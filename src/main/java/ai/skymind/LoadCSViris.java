package ai.skymind;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;

import java.io.File;
import java.util.Arrays;

public class LoadCSViris {
    private static int numLinesToSkip = 0;
    private static char delimiter = ',';

    private static int batchSize = 150; // Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
    private static int labelIndex = 4; // index of label/class column
    private static int numClasses = 3; // number of class in iris dataset

    public static void main(String[] args) throws Exception {

        // define csv file location
        File inputFile = new ClassPathResource("iris.txt").getFile();
        FileSplit fileSplit = new FileSplit(inputFile);

        // uncomment below if you wish to use bostonHousing dataset for this code >.<
        //File inputFile = new ClassPathResource("datavec/bostonHousing.csv").getFile();

        // get dataset using record reader. CSVRecordReader handles loading/parsing
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(fileSplit); //to initialize variable

        // create iterator from record reader
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
        DataSet allData = iterator.next();

        System.out.println("Shape of allData vector:"); //put all data in vector form
        System.out.println(Arrays.toString(allData.getFeatures().shape())); //output: [a,b]

        /**Shuffle**/
        // if we do not shuffle the data, training and test data set might be biased.
        // by shuffling, we can ensure the each data point has "independent" change on the model

        //System.out.println("Shuffle data:");
        allData.shuffle();  //shuffle all data

        /**Splitting data set into training and test data set**/
        // split all data into training and test set.
        // splitting is important to evaluate data mining models.
        // Analysis Services randomly samples the data to help ensure that
        // the testing and training sets are similar.
        // By using similar data for training and testing, you can minimize
        // the effects of data discrepancies and better understand the characteristics of the model.

        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);
        DataSet trainingData = testAndTrain.getTrain(); //to get train data set
        DataSet testData = testAndTrain.getTest(); //to get train data set

        System.out.println("\nShape of training vector:"); //training data set in vector form
        System.out.println(Arrays.toString(trainingData.getFeatures().shape()));

        System.out.println("\nShape of test vector:"); //test data set in vector form
        System.out.println(Arrays.toString(testData.getFeatures().shape()));

        /**Iterators**/
        // iterator is a data iteration tools for loading into neural networks.
        // a dataset iterator allows for easy loading of data into neural networks.
        // iterator helps organize batching. conversion and masking.

        // create iterator for both training and test dataset
        // how to set batch size for train iterator and test iterator??
        DataSetIterator trainIterator = new ViewIterator(trainingData, 4);
        DataSetIterator testIterator = new ViewIterator(testData, 2);

        /**Normalization**/
        // the process of rescaling one or more attributes to the range of 0 to 1.
        // to improve the performance of machine learning models.
        // 3 types (z-normalization, min-max normalization (aka rescaling), unit vector normalization).

        // normalize data to 0 - 1 (min-max normalization)
        DataNormalization scaler = new NormalizerMinMaxScaler(0,1);
        scaler.fit(trainIterator); // why we dont normalize test iterator?
        trainIterator.setPreProcessor(scaler);
        testIterator.setPreProcessor(scaler);

        System.out.println("\nShape of training batch vector:");
        System.out.println(Arrays.toString(trainIterator.next().getFeatures().shape()));
        System.out.println("\nShape of test batch vector:");
        System.out.println(Arrays.toString(testIterator.next().getFeatures().shape()));
        System.out.println("\ntraining vector: ");
        System.out.println(trainIterator.next().getFeatures());
        System.out.println("\ntest vector: ");
        System.out.println(testIterator.next().getFeatures());
    }
}
