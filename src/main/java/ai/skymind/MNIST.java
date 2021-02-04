package ai.skymind;


import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.IOException;

public class MNIST {

    // define seed, batch size and epoch
    final  static int seed = 1234;
    final static int batchSize = 500;
    final static int epoch = 1; //may vary

    // define main method
    public static void main (String[] args) throws IOException {

        MnistDataSetIterator trainMnist = new MnistDataSetIterator(batchSize, true, seed);
        MnistDataSetIterator testMnist = new MnistDataSetIterator(batchSize, false, seed);

        // normalization
        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0,1);
        //DataSet trainSet = trainMnist.next();
        //scaler.fit(trainSet); - no need this in here
        //scaler.transform(trainSet); - no need this in here
        scaler.fit(trainMnist); //only scaler fit the trainMnist,
        trainMnist.setPreProcessor(scaler);
        testMnist.setPreProcessor(scaler);

        //System.out.println(trainMnist.next().numExamples());
        //System.out.println(trainSet.numExamples());

        // model configuration
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed (seed)
                .updater(new Adam(1e-3))
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .list()

                .layer(new DenseLayer.Builder()
                        .nIn(trainMnist.inputColumns())
                        .nOut(100)
                        .build())

                .layer(new DenseLayer.Builder()
                        .nOut(100)
                        .build())

                .layer(new DenseLayer.Builder()
                        .nOut(100)
                        .build())


                .layer(new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nOut(trainMnist.totalOutcomes())
                        .build())

                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        InMemoryStatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();

        server.attach(storage);
        model.setListeners(new StatsListener(storage), new ScoreIterationListener(1000));
        //server.isRemoteListenerEnabled();

        for(int i = 0; i<= epoch ; i++){

            model.fit(trainMnist);
        }

        // evaluate model for training and test dataset
        Evaluation evalTrain = model.evaluate(trainMnist);
        Evaluation evalTest = model.evaluate(testMnist);

        // print the stats for both training and test dataset
        System.out.println("Statistical results for training set:");
        System.out.println(evalTrain.stats());
        System.out.println("Statistical results for test set:");
        System.out.println(evalTest.stats());

    }

}

