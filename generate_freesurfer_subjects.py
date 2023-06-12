import os
import numpy as np
import neuroboros as nb
import nibabel as nib


for ico in [8, 16, 32, 64, 128]:
    for lr in 'lr':
        for which in ['sphere.reg', 'pial', 'white', 'midthickness']:
            out_fn = f'subjects/onavg-ico{ico}/surf/{lr}h.{which}'
            os.makedirs(os.path.dirname(out_fn), exist_ok=True)
            coords, faces = nb.geometry(which, lr, f'onavg-ico{ico}')
            if which == 'sphere.reg':
                coords *= 100
            nib.freesurfer.write_geometry(out_fn, coords, faces)

        mask = nb.mask(lr, f'onavg-ico{ico}')
        coords = nb.geometry('midthickness', lr, f'onavg-ico{ico}', vertices_only=True)
        indices = np.where(mask)[0]
        out_fn = f'subjects/onavg-ico{ico}/label/{lr}h.cortex.label'
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        with open(out_fn, 'w') as f:
            f.write(f'#label cerebral cortex, onavg-ico{ico}\n')
            f.write(f'{len(indices)}\n')
            for i in indices:
                x, y, z = coords[i]
                f.write(f'{i} {x:.3f} {y:.3f} {z:.3f} 0\n')
