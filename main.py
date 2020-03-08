from model import *
from data import *
import time

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

# model = unet()
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
# model.fit_generator(myGene,steps_per_epoch=5,epochs=1,callbacks=[model_checkpoint])
# model.save('unet_model.h5')
model = load_model('unet_model.h5')

model.summary()
testGene = testGenerator("data/membrane/test")
start = time.time()
results = model.predict_generator(testGene,1,verbose=1)
end = time.time()
print("time:",end-start)
saveResult("img/test_result",results)



results = results[0]
for i in range(results.shape[0]):
    for j in range(results.shape[1]):
        # if results[i][j][0]<0.1:
        print(results[i][j][0])
# print(results)

