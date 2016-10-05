/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machine_learing_clasifier;

import java.util.Enumeration;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.Id3;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author yoga
 */
public class MyID3 extends AbstractClassifier{
    
    /** The node's successors. */ 
    private MyID3[] m_Successors;

    /** Attribute used for splitting. */
    private Attribute m_Attribute;

    /** Class value if node is leaf. */
    private double m_ClassValue;

    /** Class distribution if node is leaf. */
    private double[] m_Distribution;

    /** Class attribute of dataset. */
    private Attribute m_ClassAttribute;
    
    public MyID3() {
    
    }

    @Override
    public void buildClassifier(Instances i) throws Exception {
        if (!i.classAttribute().isNominal()) {
            throw new Exception("Class not nominal");
        }
        
        for (int j = 0; j < i.numAttributes(); j++) {
            Attribute attr = i.attribute(j);
            if (!attr.isNominal()) {
                throw new Exception("Attribute not nominal");
            }
            
            for (int k = 0; k < i.numInstances(); k++) {
                Instance inst = i.instance(k);
                if (inst.isMissing(attr)) {
                    throw new Exception("Missing value");
                }
            }
        }
        
        i = new Instances(i);
        i.deleteWithMissingClass();
        makeTree(i);
    }
    
    public double computeEntropy(Instances inst) {
        double[] classCount = new double[inst.numClasses()];
        for (int i = 0; i < inst.numInstances(); i++) {
            int temp = (int) inst.instance(i).classValue();
            classCount[temp]++;
        }
        double entropy = 0;
        for (int i = 0; i < inst.numClasses(); i++) {
            if (classCount[i] > 0) {
                entropy -= classCount[i] * Utils.log2(classCount[i]/inst.numInstances());
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

        for (int i = 0; i < inst.numInstances(); i++) {
            int temp = (int) inst.instance(i).value(attr);
            split[temp].add(inst.instance(i));
        }
        
        return split;
    }
    
    public void makeTree(Instances data) throws Exception{
        if (data.numInstances() == 0) {
          return;
        }
        
        double[] infoGains = new double[data.numAttributes()];
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute att = data.attribute(i);
            if (data.classIndex() != att.index()) {
                infoGains[att.index()] = computeInformationGain(data, att);
            }
        }
        
        m_Attribute = data.attribute(Utils.maxIndex(infoGains));
        System.out.println("huhu = " + m_Attribute.toString());
        
        if (Utils.eq(infoGains[m_Attribute.index()], 0)) {
          m_Attribute = null;
          m_Distribution = new double[data.numClasses()];
          for (int i = 0; i < data.numInstances(); i++) {
                int inst = (int) data.instance(i).value(data.classAttribute());
                m_Distribution[inst]++;
          }
          Utils.normalize(m_Distribution);
          m_ClassValue = Utils.maxIndex(m_Distribution);
          m_ClassAttribute = data.classAttribute();
        } else {
          Instances[] splitData = splitData(data, m_Attribute);
          m_Successors = new MyID3[m_Attribute.numValues()];
          for (int j = 0; j < m_Attribute.numValues(); j++) {
            m_Successors[j] = new MyID3();
            m_Successors[j].buildClassifier(splitData[j]);
          }
        }
    }
    
    public double classifyInstance(Instance instance){
        if(m_Attribute == null){
            return m_ClassValue;
        }
        else{
            return m_Successors[(int) instance.value(m_Attribute)].classifyInstance(instance);
        }
    }
}
