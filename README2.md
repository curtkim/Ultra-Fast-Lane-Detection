## HOWTO

    mkdir checkout
    # download tusimple_18.pth

    mkdir tmp
    python demo_interactive.py configs/tusimple.py --test_model checkout/tusimple_18.pth 
    ffmpeg -framerate 30 -i 'tmp/frame%04d.jpg' demo.mp4
