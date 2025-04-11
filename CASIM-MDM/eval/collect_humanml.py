
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.parser_util import evaluation_parser, generate_args
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader  # get_motion_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from collections import OrderedDict
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from utils.model_util import create_model_and_diffusion, load_model_wo_clip

from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    args = evaluation_parser()
    args.save_attn_value = True

    fixseed(args.seed)
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_file = os.path.join(os.path.dirname(args.model_path), 'eval_humanml_{}_{}'.format(name, niter))
    if args.guidance_param != 1.:
        log_file += f'_gscale{args.guidance_param}'
    log_file += f'_{args.eval_mode}'
    log_file += '.log'

    print(f'Will save to log file [{log_file}]')

    print(f'Eval mode [{args.eval_mode}]')
    
    num_samples_limit = None
    run_mm = False
    mm_num_samples = 0
    mm_num_repeats = 0
    mm_num_times = 0
    diversity_times = 300
    replication_times = 1

    dist_util.setup_dist(args.device)

    split = 'test'
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='eval')

    model, diffusion = create_model_and_diffusion(args, gen_loader)

    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    
    scale = args.guidance_param
    use_ddim = False  # FIXME - hardcoded
    clip_denoised = False  # FIXME - hardcoded
    sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )
    
    all_motions = []
    all_lengths = []
    all_text = []
    
    if args.save_attn_value:
        all_attention_values = []
        all_attention_masks = []
    
    with torch.no_grad():
        for i, (motion, model_kwargs) in tqdm(enumerate(gen_loader)):
            n_frames = motion.shape[-1]
            tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

            # add CFG scale to batch
            if scale != 1.:
                model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                        device=dist_util.dev()) * scale
            sample = sample_fn(
                model,
                motion.shape,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
                # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
            )
            if args.save_attn_value:
                attn_value = torch.stack(model.model.saved_attention_values, dim=0).cpu().numpy() # [diffusion_steps, batch_size, n_frames, n_token]
                all_attention_values.append(attn_value)
                attn_mask = torch.stack(model.model.saved_attention_masks, dim=0).cpu().numpy()
                all_attention_masks.append(attn_mask)
                model.model.clear_attention_cache()
                
            # Recover XYZ *positions* from HumanML3D vector representation
            if model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = gen_loader.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
            rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
            sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                get_rotations_back=False)
            
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]
            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
            batch_save = {
                'motion': sample.cpu().numpy(),
                'lengths': model_kwargs['y']['lengths'].cpu().numpy(),
                'text': model_kwargs['y'][text_key],
            }
            if args.save_attn_value:
                batch_save['attention_values'] = attn_value
                batch_save['attention_masks'] = attn_mask
            # Save to file
            output_dir = os.path.join('save', 'testset_output')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{name}_batch_{i}.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(batch_save, f)
                

            
    # all_motions = np.concatenate(all_motions, axis=0)
    # all_motions = all_motions  # [bs, njoints, 6, seqlen]
    # all_text = all_text
    # all_lengths = np.concatenate(all_lengths, axis=0)
    # if args.save_attn_value:
    #     all_attention_values = np.concatenate(all_attention_values, axis=1)
    #     all_attention_masks = np.concatenate(all_attention_masks, axis=1) # [diffusion_steps, batch_size,...]
        
    # # Save to file
    # output_dir = os.path.join('save', 'testset_output')
    # os.makedirs(output_dir, exist_ok=True)
    
    # output_dict = {
    #     'motions': all_motions,
    #     'lengths': all_lengths,
    #     'text': all_text,
    # }
    # if args.save_attn_value:
    #     output_dict['attention_values'] = all_attention_values
    #     output_dict['attention_masks'] = all_attention_masks
    # output_file = os.path.join(output_dir, f'{name}_{niter}_{args.eval_mode}.pkl')
    # with open(output_file, 'wb') as f:
    #     pickle.dump(output_dict, f)
    # print(f'Saved to [{output_file}]')
    
    

