import coremltools
import coremltools.proto.FeatureTypes_pb2 as ft

def convert_multiarray_output_to_image(spec, feature_name, is_bgr=False):
    """
    Convert an output multiarray to be represented as an image
    This will modify the Model_pb spec passed in.
    Example:
        model = coremltools.models.MLModel('MyNeuralNetwork.mlmodel')
        spec = model.get_spec()
        convert_multiarray_output_to_image(spec,'imageOutput',is_bgr=False)
        newModel = coremltools.models.MLModel(spec)
        newModel.save('MyNeuralNetworkWithImageOutput.mlmodel')
    Parameters
    ----------
    spec: Model_pb
        The specification containing the output feature to convert
    feature_name: str
        The name of the multiarray output feature you want to convert
    is_bgr: boolean
        If multiarray has 3 channels, set to True for RGB pixel order or false for BGR
    """
    for output in spec.description.output:
        if output.name != feature_name:
            continue
        if output.type.WhichOneof('Type') != 'multiArrayType':
            raise ValueError("%s is not a multiarray type" % output.name)
        array_shape = tuple(output.type.multiArrayType.shape)
        channels, height, width = array_shape
        from coremltools.proto import FeatureTypes_pb2 as ft
        if channels == 1:
            output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
        elif channels == 3:
            if is_bgr:
                output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('BGR')
            else:
                output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('RGB')
        else:
            raise ValueError("Channel Value %d not supported for image inputs" % channels)
        output.type.imageType.width = width
        output.type.imageType.height = height

mlmodel = coremltools.converters.keras.convert('unet_model.h5', image_scale=1.0/255, input_names="input_layer", image_input_names='input_layer')
# mlmodel = coremltools.converters.keras.convert('unet_model.h5')
spec = mlmodel.get_spec()

output = spec.description.output[0]
# output.type.multiArrayType.dataType = ft.ArrayFeatureType.FLOAT32


# convert_multiarray_output_to_image(spec,'output1',is_bgr=False)
mlmodel = coremltools.models.MLModel(spec)
mlmodel.save('unet_model.mlmodel')



# print(mlmodel.get_spec())
# print(spec.description.output)
# output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
# output.type.imageType.height = 256
# output.type.imageType.width = 256
# # print(spec.description.output[0])
# mlmodel.save('unet_model.mlmodel')

# # convert by providing a Keras model object
# from keras.models import load_model
# keras_model = load_model("keras_model.h5")
# mlmodel = coremltools.converters.keras.convert(keras_model)


# spec = coremltools.utils.load_spec("unet_model.mlmodel")
#
# input = spec.description.input[0]
# input.type.imageType.colorSpace = ft.ImageFeatureType.
# input.type.imageType.height = 256
# input.type.imageType.width = 256
#
# coremltools.utils.save_spec(spec, "processed_unet_model.mlmodel")