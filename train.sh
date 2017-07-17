#python main.py --arch resnet50 --classes 12 --bottleneck 256 --lr 0.00003 --pretrained --test-iter 1000 /home/liushichen/playground/taskcv-2017-public/classification/data --gpu $1 --batch-size 64 --print-freq 100 --model jan

#python main.py --arch alexnet --classes 12 --bottleneck 256 --lr 0.0001 --pretrained --test-iter 1000 /home/liushichen/playground/taskcv-2017-public/classification/data --gpu $1 --batch-size 64 --print-freq 100 --model jan

#python main.py --arch alexnet --classes 12 --bottleneck 128 --lr 0.0003 --pretrained --test-iter 1000 /home/liushichen/playground/taskcv-2017-public/classification/data --gpu $1 --batch-size 64 --print-freq 100

### 07-13
### 63.5++
python main.py --arch resnet152 --classes 12 --bottleneck 256 --lr 0.0001 --pretrained --test-iter 1000 /home/liushichen/playground/taskcv-2017-public/classification/data --batch-size 32 --print-freq 100 --model jan --gpu 

###
python main.py --arch resnet101 --classes 12 --bottleneck 256 --lr 0.0001 --pretrained --test-iter 1000 /home/liushichen/playground/taskcv-2017-public/classification/data --batch-size 32 --print-freq 100 --model dan --gpu 
