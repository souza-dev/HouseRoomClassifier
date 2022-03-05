import React, { useState, useEffect} from 'react';
import { Text, View, Platform, ActivityIndicator, useWindowDimensions } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import { bundleResourceIO, cameraWithTensors } from '@tensorflow/tfjs-react-native';
import { Camera } from 'expo-camera';

const TensorCamera = cameraWithTensors(Camera);

export default function App() {

  let requestAnimationFrameId = 0; 
  const { height, width } = useWindowDimensions();

  const [hasPermission, setHasPermission] = useState(null);
  const [isTfReady, setIsTfReady] = useState(false);
  const [loading, setIsLoading] = useState(true);
  const [prediction, setPrediction] = useState(null);
  const [model, setModel] = useState(null);
  
  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  useEffect(() => {
    async function loadingTf() {
      console.log('Carregando tensor flow');
      setIsLoading(true);
      await tf.ready();
      setIsTfReady(true);
      console.log('TF carregado');
    }
    loadingTf();
  }, []);

  useEffect(() => {
    loadMyModel()
  }, []);

  useEffect(() => {
    return () => {
      cancelAnimationFrame(requestAnimationFrameId);
    };
  }, [requestAnimationFrameId]);
  
  const loadMyModel = async () => {
      const modelJson = require('./assets/model/model.json');
      const modelWeights = require('./assets/model/group1-shard1of1.bin');
      const model = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
      setModel(model);
      setIsLoading(false);
  }

  const doPredict = async (imageTensor) => {

    if (!imageTensor || !model) { return; }

    const tensor = imageTensor.expandDims(0).toFloat()
    const prediction = await model.predict(tensor).array();
    const max = Math.max(...prediction[0]);
    const index = prediction[0].indexOf(max);

    switch(index){
      case 0:
        setPrediction('Banheiro');
        break;
      case 1:
        setPrediction('Quarto');
        break;
      case 2:
        setPrediction('Sala de jantar');
        break;
      case 3:
        setPrediction('Cozinha');
        break;
      case 4:
        setPrediction('Sala de estar');
        break;
      default:
        setPrediction('Não definido')
    }
  }

  let textureDims;
  if (Platform.OS === 'ios') {
   textureDims = {
     height: 1920,
     width: 1080,
   };
  } else {
   textureDims = {
     height: 1200,
     width: 1600,
   };
  }

  const handleCameraStream = (images) => {
    const loop = async () => {
      const nextImageTensor = images.next().value;
      doPredict(nextImageTensor);
      requestAnimationFrameId = requestAnimationFrame(loop);
    };
    loop();
  } 

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }

  if (loading) {
    return (
      <View style={{flex: 1, justifyContent: 'center', alignItems: 'center'}}>
        <ActivityIndicator size="large" animating={true}  />
        <Text>{'Carregando'}</Text>
      </View>
    )
  }

  return (
      <View style={{justifyContent: 'center', alignItems: 'center', flex: 1}}>
        { loading && <ActivityIndicator size="large" animating={true}  />}
        {
          model && (
            <View>
              <TensorCamera 
                style={{height: height - 100, width: width}} 
                type={Camera.Constants.Type.back} 
                cameraTextureHeight={textureDims.height}
                cameraTextureWidth={textureDims.width}
                resizeHeight={224}
                resizeWidth={224}
                resizeDepth={3}
                onReady={handleCameraStream}
                autorender={true}
                useCustomShadersToResize={false}
              />
            <Text style={{fontSize: 25, color: 'red', textAlign: 'center'}}>{'Predição: '+prediction}</Text> 
            </View>

          )
        }
      </View>
  );
}