/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package function;

import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

/**
 *
 * @author ginanjarbusiri
 */
public class CrossValidation {
    public static void crossValidation(Instances data, AbstractClassifier cls) throws Exception {
        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(cls, data, 10, new Random(1));
        System.out.println(evaluation.toSummaryString());
    }
}
