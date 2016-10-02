/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package function;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;

/**
 *
 * @author yoga
 */
public class FileModel {

    public static void SaveModel(String sourcepath, String outputpath) throws IOException, Exception {
        // create J48

        //kayanya sih ntar ganti sama class clasifiernya
        Classifier cls = new J48();

        // train
        Instances inst = new Instances(
                new BufferedReader(
                        new FileReader(sourcepath)));
        inst.setClassIndex(inst.numAttributes() - 1);
        cls.buildClassifier(inst);

        // serialize model
        weka.core.SerializationHelper.write(outputpath, cls);
    }

    public static void LoadModel(String sourcepath) throws Exception {
        // deserialize model
        Classifier cls = (Classifier) weka.core.SerializationHelper.read(sourcepath);
    }
}
