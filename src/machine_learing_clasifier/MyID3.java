/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machine_learing_clasifier;

import java.util.Enumeration;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author yoga
 */
public class MyID3 extends AbstractClassifier{

    @Override
    public void buildClassifier(Instances i) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public double computeEntropy(Instances inst) {
        double[] classCount = new double[inst.numClasses()];
        Enumeration instEnum = inst.enumerateInstances();
        while(instEnum.hasMoreElements()) {
            Instance temp = (Instance) instEnum.nextElement();
            classCount[(int) temp.classValue()]++;
        }
        double entropy = 0;
        for (int i = 0; i < inst.numClasses(); i++) {
            if (classCount[i] > 0) {
                entropy -= classCount[i] * Utils.log2(classCount[i]/classCount[inst.numInstances()]);
            }
        }
        entropy /= (double) inst.numInstances();
        return entropy;
    }
    
    public double computeInformationGain(Instances inst, Attribute attr) {
        double gain = computeEntropy(inst);
        Instances[] split = splitData(inst, attr);
        for (int i = 0; i < attr.numValues(); i++) {
            if (split[i].numInstances() > 0) {
                gain -= ((double) split[i].numInstances() / (double) inst.numInstances()) * computeEntropy(split[i]);
            }
        }
        
        return gain;
    }
    
    public Instances[] splitData(Instances inst, Attribute attr) {
        Instances[] split = new Instances[attr.numValues()];
        for (int i = 0; i < attr.numValues(); i++) {
            split[i] = new Instances(inst, inst.numInstances());
        }
        Enumeration instEnum = inst.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance temp = (Instance) instEnum.nextElement();
            split[(int) temp.value(attr)].add(temp);
        }
        
        return split;
    }
}
