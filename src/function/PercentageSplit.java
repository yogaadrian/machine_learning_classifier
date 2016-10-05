/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package function;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import weka.core.Instances;
/**
 *
 * @author ginanjarbusiri
 */
public class PercentageSplit {
    public static void percentageSplit(Instances data, Classifier cls) throws Exception {
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(cls, test);
        System.out.println(eval.toSummaryString());
    }
    
    public static double percentageSplitRate(Instances data, Classifier cls) throws Exception {
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(cls, test);
        return eval.pctCorrect();
    }
}
