/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package function;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

/**
 *
 * @author ginanjarbusiri
 */
public class FilterResample {
    public static Instances filterResample(Instances inst) {
        Resample filter = new Resample();
        Instances instResample = null;
        filter.setBiasToUniformClass(1.0);
        try {
            filter.setInputFormat(inst);
            filter.setNoReplacement(false);
            filter.setSampleSizePercent(100);
            instResample = Filter.useFilter(inst, filter);
        } catch(Exception e) {
            System.out.println("Error when resampling input data!");
            e.printStackTrace();
        }
        
        return instResample;       
    }
}
