/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machine_learing_clasifier;

import function.PercentageSplit;
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
public class MyC45 extends AbstractClassifier {

    /**
     * The node's successors.
     */
    private MyC45[] m_Successors;

    /**
     * Attribute used for splitting.
     */
    private Attribute m_Attribute;

    /**
     * Class value if node is leaf.
     */
    private double m_ClassValue;

    /**
     * Class distribution if node is leaf.
     */
    private double[] m_Distribution;

    /**
     * Class attribute of dataset.
     */
    private Attribute m_ClassAttribute;

    private double numericAttThreshold;

    public MyC45 head, parent;

    public MyC45() {
        head = this;
    }

    public MyC45(MyC45 head, MyC45 parent) {
        this.head = head;
        this.parent = parent;
    }

    @Override
    public void buildClassifier(Instances i) throws Exception {
        if (!i.classAttribute().isNominal()) {
            throw new Exception("Class not nominal");
        }

        //penanganan missing value
        for (int j = 0; j < i.numAttributes(); j++) {
            Attribute attr = i.attribute(j);
            for (int k = 0; k < i.numInstances(); k++) {
                Instance inst = i.instance(k);
                if (inst.isMissing(attr)) {
                    inst.setValue(attr, fillMissingValue(i, attr));
                    //bisa dituning lagi performancenya
                }
            }
        }

        i = new Instances(i);
        i.deleteWithMissingClass();
        makeTree(i);
    }

    public double classifyInstance(Instance instance) {
        if (m_Attribute == null) {
            return m_ClassValue;
        } else {
            if (m_Attribute.isNominal()) {
                return m_Successors[(int) instance.value(m_Attribute)].classifyInstance(instance);
            } else if (m_Attribute.isNumeric()) {
                if (instance.value(m_Attribute) < numericAttThreshold) {
                    return m_Successors[0].classifyInstance(instance);
                } else {
                    return m_Successors[1].classifyInstance(instance);
                }
            } else {
                return -1;
            }
        }
    }

    public void prune(Instances i) throws Exception {
        if (m_Successors != null) {
            System.out.println("test");
            for (int a = 0; a < m_Successors.length; a++) {
                m_Successors[a].prune(i);
                calculateErrorPrune(i, a);
                break;
            }
        }
    }

    public void calculateErrorPrune(Instances i, int order) throws Exception {
        double before, after;
        before = PercentageSplit.percentageSplitRate(i, head);
        System.out.println("Order " +order);
        MyC45 temp = this.parent.m_Successors[order];
        this.parent.m_Successors[order] = null;
        after = PercentageSplit.percentageSplitRate(i, head);
        System.out.println("after " + after);
        System.out.println("before" + before);
        System.out.println("");
        if (before < after) {
            this.parent.m_Successors[order] = temp;
        } else {
            System.out.println("prune!!!");
        }
    }

    public double fillMissingValue(Instances i, Attribute att) {
        int[] jumlahvalue = new int[att.numValues()];
        for (int k = 0; k < i.numInstances(); k++) {
            jumlahvalue[(int) i.instance(k).value(att)]++;
        }
        return jumlahvalue[Utils.maxIndex(jumlahvalue)];
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
                entropy -= classCount[i] * Utils.log2(classCount[i] / inst.numInstances());
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

    public double computeInformationGainContinous(Instances inst, Attribute attr, double threshold) {
        double gain = computeEntropy(inst);
        Instances[] split = splitDataContinous(inst, attr, threshold);
        for (int i = 0; i < 2; i++) {
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

    public Instances[] splitDataContinous(Instances inst, Attribute attr, double threshold) {
        Instances[] split = new Instances[2];
        for (int i = 0; i < 2; i++) {
            split[i] = new Instances(inst, inst.numInstances());
        }

        for (int i = 0; i < inst.numInstances(); i++) {
            double temp = inst.instance(i).value(attr);
            if (temp < threshold) {
                split[0].add(inst.instance(i));
            } else {
                split[1].add(inst.instance(i));
            }
        }

        return split;
    }

    public void makeTree(Instances data) throws Exception {
        if (data.numInstances() == 0) {
            return;
        }

        double[] infoGains = new double[data.numAttributes()];
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute att = data.attribute(i);
            if (data.classIndex() != att.index()) {
                if (att.isNominal()) {
                    infoGains[att.index()] = computeInformationGain(data, att);
                } else {
                    infoGains[att.index()] = computeInformationGainContinous(data, att, BestContinousAttribute(data, att));
                }
            }
        }

        m_Attribute = data.attribute(Utils.maxIndex(infoGains));
        if (m_Attribute.isNumeric()) {
            numericAttThreshold = BestContinousAttribute(data, m_Attribute);
            System.out.println(" ini kalo continous dengan attribut : " + numericAttThreshold);
        }
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
            Instances[] splitData;
            if (m_Attribute.isNominal()) {
                splitData = splitData(data, m_Attribute);
            } else {
                splitData = splitDataContinous(data, m_Attribute, numericAttThreshold);
            }

            if (m_Attribute.isNominal()) {
                System.out.println("nominal");
                m_Successors = new MyC45[m_Attribute.numValues()];
                System.out.println(m_Successors.length);
                for (int j = 0; j < m_Attribute.numValues(); j++) {
                    m_Successors[j] = new MyC45(head, this);
                    m_Successors[j].buildClassifier(splitData[j]);
                }
            } else {
                System.out.println("numeric");
                m_Successors = new MyC45[2];
                System.out.println(m_Successors.length);
                for (int j = 0; j < 2; j++) {
                    m_Successors[j] = new MyC45(head, this);
                    m_Successors[j].buildClassifier(splitData[j]);
                }
            }
        }
    }

    public double BestContinousAttribute(Instances i, Attribute att) {

        i.sort(att);
        Enumeration enumForMissingAttr = i.enumerateInstances();
        double temp = i.get(0).classValue();
        double igtemp = 0;
        double bestthreshold = 0;
        double a;
        double b = i.get(0).value(att);
        while (enumForMissingAttr.hasMoreElements()) {
            Instance inst = (Instance) enumForMissingAttr.nextElement();
            if (temp != inst.classValue()) {
                temp = inst.classValue();
                a = b;
                b = inst.value(att);
                double threshold = a + ((b - a) / 2);
                double igtemp2 = computeInformationGainContinous(i, att, threshold);
                if (igtemp < igtemp2) {
                    bestthreshold = threshold;
                    igtemp = igtemp2;
                }

            }

        }
        return bestthreshold;
    }

}
