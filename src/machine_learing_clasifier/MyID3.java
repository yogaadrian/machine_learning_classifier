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
    private Id3[] m_Successors;

    /** Attribute used for splitting. */
    private Attribute m_Attribute;

    /** Class value if node is leaf. */
    private double m_ClassValue;

    /** Class distribution if node is leaf. */
    private double[] m_Distribution;

    /** Class attribute of dataset. */
    private Attribute m_ClassAttribute;


    @Override
    public void buildClassifier(Instances i) throws Exception {
        if (!i.classAttribute().isNominal()) {
            throw new Exception("Class not nominal");
        }
        
        Enumeration enumAttr = i.enumerateAttributes();
        while(enumAttr.hasMoreElements()) {
            Attribute attr = (Attribute) enumAttr.nextElement();
            if (!attr.isNominal()) {
                throw new Exception("Attribute not nominal");
            }
            Enumeration enumForMissingAttr = i.enumerateInstances();
            while(enumForMissingAttr.hasMoreElements()) {
                if (((Instance) enumForMissingAttr.nextElement()).isMissing(attr)) {
                    throw new Exception("Missing value");
                }
            }
        }
        
        i = new Instances(i);
        i.deleteWithMissingClass();
        //makeTree(i);
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
    
    public void makeTree(Instances data) throws Exception{
        // Check if no instances have reached this node.
        if (data.numInstances() == 0) {
          return;
        }

        // Compute attribute with maximum information gain.
        double[] infoGains = new double[data.numAttributes()];
        Enumeration attEnum = data.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
          Attribute att = (Attribute) attEnum.nextElement();
          infoGains[att.index()] = computeInformationGain(data, att);
        }
        m_Attribute = data.attribute(Utils.maxIndex(infoGains));

        // Make leaf if information gain is zero. 
        // Otherwise create successors.
        if (Utils.eq(infoGains[m_Attribute.index()], 0)) {
          m_Attribute = null;
          m_Distribution = new double[data.numClasses()];
          Enumeration instEnum = data.enumerateInstances();
          while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            m_Distribution[(int) inst.classValue()]++;
          }
          Utils.normalize(m_Distribution);
          m_ClassValue = Utils.maxIndex(m_Distribution);
          m_ClassAttribute = data.classAttribute();
        } else {
          Instances[] splitData = splitData(data, m_Attribute);
          m_Successors = new Id3[m_Attribute.numValues()];
          for (int j = 0; j < m_Attribute.numValues(); j++) {
            m_Successors[j] = new Id3();
            m_Successors[j].buildClassifier(splitData[j]);
          }
        }
        
    }
}
