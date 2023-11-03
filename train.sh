export CXX="g++"
python -m pdb train.py \
	--batchSize 2 \
	--nThreads 2 \
	--name comod_frame_synthesis \
	--load_size 256 \
	--crop_size 256 \
	--z_dim 512 \
	--validation_freq 10000 \
	--niter 50 \
	--dataset_mode splatting \
	--trainer stylegan2 \
	--dataset_mode_train splatting \
	--dataset_mode_val splatting \
	--model comod \
	--netG comodgan \
	--netD comodgan \
	--no_l1_loss \
	--freeze_D \
	--use_encoder \
	--verbose \
	--no_vgg_loss \
	--preprocess_mode scale_shortside_and_crop \
	$EXTRA
