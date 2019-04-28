#! /bin/bash

#cd ~/project/video_processing
#python process.py
#cp -r images/* ~/project/smplify_test/images/

cd ~/project/smplify_test/code/deepcut-cnn/python/pose/
python pose_demo.py ../../../../images --out_name=../../../../results/prediction --use_cpu

cd ../../../
python fit_3d.py female ../ --out_dir=./pkl

python load_pkl.py

cd contourForWrap/
python image_position.py

cd ..
python printOBJ/render_smpl.py

