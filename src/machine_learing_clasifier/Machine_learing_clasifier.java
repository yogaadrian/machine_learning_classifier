/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */



package machine_learing_clasifier;
import function.LoadData;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author yoga
 */
public class Machine_learing_clasifier {
    public static MyC45 myc45 = new MyC45();
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        Instances data = new Instances(LoadData.getData("D:\\Documents\\ML\\machine_learing_clasifier\\weather.numeric.arff"));
        myc45.buildClassifier(data);
        
    }

    
}
