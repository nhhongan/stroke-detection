package Evaluation;

import static java.lang.Math.random;

import java.util.Random;

import Data_processing.Pre_process_base;
import Model.WekaModel;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;

public class KCrossVal {
    public WekaModel model;
    public Instances data;
    public int folds;
    public Pre_process_base pre_process;

    public KCrossVal(WekaModel model, Instances data, int folds){
        this.model = model;
        this.pre_process = new Pre_process_base();
        this.data = data;
        this.folds = folds;
    }

    public void k_folds_validation() throws Exception {
        WekaModel copy_model = model.copy();
        copy_model.buildClassifier(data);
        validation(copy_model, data);
    }

    // public double printInformation(double[] f1Scores, double folds) {
    //     double avgF1Score = Utils.mean(f1Scores);
    //     System.out.println("Average F1 Score: " + avgF1Score);
    //     return avgF1Score;
    // }

    public void validation(WekaModel model, Instances data) {
        try {
            Evaluation eval = new Evaluation(data);
            Random random = new Random(1);
            eval.crossValidateModel(model.classifier, data, folds, random);
            System.out.println("Avg F1 score: "+eval.fMeasure(1));
            System.out.println("Precision: "+eval.precision(1));
            System.out.println("Recall: "+eval.recall(1));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
