/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */



package machine_learing_clasifier;
import function.CrossValidation;
import function.LoadData;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author yoga
 */
public class Machine_learing_clasifier {
    public static MyC45 myc45 = new MyC45();
    public static MyID3 myid3 = new MyID3();
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here

        Instances data = new Instances(LoadData.getData("/home/ginanjarbusiri/Documents/Machine Learning/machine_learning_classifier/weather.nominal.arff"));
//        myc45.buildClassifier(data);
        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(myid3, data, 10, new Random(1));
        System.out.println(evaluation.toSummaryString());
        

//        Instances data = new Instances(LoadData.getData("E:\\machine_learning_classifier\\weather.nominal.arff"));
//        myid3.buildClassifier(data);
//        for(int i = 0; i < data.size(); i++){
//            double sesuatu = myid3.classifyInstance(data.get(i));
//            System.out.println("output = "+sesuatu);
//        }

    }
    
}
