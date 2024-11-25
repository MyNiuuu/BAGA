import os
import json
import numpy as np
from tqdm import tqdm


def save_camera_data(camera_data, output_folder):

    for i, (K, D, R, T) in enumerate(zip(camera_data['K'], camera_data['D'], camera_data['R'], camera_data['T'])):
        camera_json = {}

        if K is not None:
            camera_json['K'] = K.tolist()
        if R is not None:
            camera_json['R'] = R.tolist()
        if T is not None:
            camera_json['T'] = T.tolist()
        if D is not None:
            camera_json['dist'] = D.tolist()

        file_name = f'{str(i).zfill(2)}.json'
        output_path = os.path.join(output_folder, file_name)
        with open(output_path, 'w') as f:
            json.dump(camera_json, f, indent=2)




if __name__ == '__main__':


    scenes = ["my_377", "my_386", "my_387", "my_392", "my_393", "my_394"]


    for scene in tqdm(scenes):

        source_dir = f"./data/ZJU-MoCap-Refine/{scene}"
        target_dir = f"./data/BlurZJU/sharp/{scene}"

        os.makedirs(os.path.join(target_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'mask'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'smpl_params'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'smpl_vertices'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'camera_extris'), exist_ok=True)

        os.system(f"cp -r {os.path.join(source_dir, 'images/*')} {os.path.join(target_dir, 'images')}")
        os.system(f"cp -r {os.path.join(source_dir, 'mask/*')} {os.path.join(target_dir, 'mask')}")
        os.system(f"cp -r {os.path.join(source_dir, 'smpl_params/*')} {os.path.join(target_dir, 'smpl_params')}")
        os.system(f"cp -r {os.path.join(source_dir, 'smpl_vertices/*')} {os.path.join(target_dir, 'smpl_vertices')}")
        os.system(f"cp -r {os.path.join(source_dir, 'annots.npy')} {target_dir}")

        annots = np.load(os.path.join(source_dir, 'annots.npy'), allow_pickle=True).item()
        cams = annots['cams']
        save_camera_data(cams, os.path.join(target_dir, 'camera_extris'))

        # assert False