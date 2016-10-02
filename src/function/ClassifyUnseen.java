/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package function;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.core.Instance;
/**
 *
 * @author yoga
 */
public class ClassifyUnseen {
    public static double classify(Classifier cl,Instance inst) throws Exception{
        return cl.classifyInstance(inst);
    }
}
