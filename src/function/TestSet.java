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
public class TestSet {
    public static void testSet(Instances train, Instances test, Classifier cls) throws Exception {
        cls.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(cls, test);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }
}
