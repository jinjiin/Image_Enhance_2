nohup python train_model.py model=sony dped_dir=dped/ vgg_dir=vgg_pretrained/imagenet-vgg-verydeep-19.mat >sony_30000.log &
nohup python train_model.py model=iphone dped_dir=../Image_Enhance/dped/ vgg_dir=../Image_Enhance/vgg_pretrained/imagenet-vgg-verydeep-19.mat >iphone.log &
nohup ./python ../Image_Enhance_2/train_model.py model=iphone dped_dir=../Image_Enhance/dped/ vgg_dir=../Image_Enhance/vgg_pretrained/imagenet-vgg-verydeep-19.mat >iphone.log &

nohup ./python ../Image_Enhance/train_model.py model=sony dped_dir=../Image_Enhance/dped/ vgg_dir=../Image_Enhance/vgg_pretrained/imagenet-vgg-verydeep-19.mat >sony_30000.log &

