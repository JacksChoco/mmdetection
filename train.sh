python3 tools/train.py \
    configs/htc/htc_r50_fpn_1x_coco.py \
    --work-dir train

    pip3 install -r requirements/build.txt
pip3 install -v -e .
pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch2.0.0/index.html