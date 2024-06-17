# climate projection
python src/climax/climate_projection/train.py --config configs/climate_projection.yaml --trainer.strategy=ddp --trainer.devices=1 --trainer.max_epochs=50 --data.root_dir=/home/as26840@ens.ad.etsmtl.ca/data/climate_bench/5.625deg --data.out_variables="tas" --data.batch_size=8 --model.pretrained_path='https://huggingface.co/tungnd/climax/resolve/main/5.625deg.ckpt' --model.lr=5e-4 --model.beta_1="0.9" --model.beta_2="0.99" --model.weight_decay=1e-5

# regional forecast
python src/climax/regional_forecast/train.py --config configs/regional_forecast_climax.yaml --trainer.strategy=ddp --trainer.devices=2 --trainer.max_epochs=50 --data.root_dir=/mnt/data/5.625deg_npz --data.region="NorthAmerica" --data.predict_range=72 --data.batch_size=16 --model.pretrained_path='https://huggingface.co/tungnd/climax/resolve/main/1.40625deg.ckpt' --model.lr=5e-7 --model.beta_1="0.9" --model.beta_2="0.99" --model.weight_decay=1e-5

# global forecast
python src/climax/global_forecast/train.py --config configs/global_forecast_climax.yaml --trainer.strategy=ddp --trainer.devices=1 --trainer.max_epochs=50 --data.root_dir=/home/as26840@ens.ad.etsmtl.ca/data/ERA5_np_shards --data.predict_range=72 --data.batch_size=16 --model.pretrained_path='https://huggingface.co/tungnd/climax/resolve/main/5.625deg.ckpt' --model.lr=5e-7 --model.beta_1="0.9" --model.beta_2="0.99" --model.weight_decay=1e-5

# wildfire
CUDA_VISIBLE_DEVICES=1 python src/climax/wildfireTS/train.py --config configs/wildfirespreadTS.yaml --trainer.strategy=ddp --trainer.devices=1 --trainer.max_epochs=50 --data.root_dir=/home/as26840@ens.ad.etsmtl.ca/data/wildfirespreadTS_hdf5 --data.predict_range=24 --data.batch_size=2 --model.pretrained_path='https://huggingface.co/tungnd/climax/resolve/main/1.40625deg.ckpt' --model.lr=5e-7 --model.beta_1="0.9" --model.beta_2="0.99" --model.weight_decay=1e-5