package function;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author ginanjarbusiri
 */
public class RemoveAtribut {
    
    public static Instances removeAttribut(Instances inst) throws IOException, Exception {
        Instances ret = null;
        Remove rem = new Remove();
        
        if (inst.classIndex() < 0) {
            ret = inst;
        } else {
            rem.setAttributeIndices(""+(inst.classIndex()+1));
            rem.setInvertSelection(false);
            rem.setInputFormat(inst);
            ret = Filter.useFilter(inst, rem);
        }
            
        return ret;
    }
        
}
