package Evaluation;

import Model.WekaModel;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class KCrossVal {
    public WekaModel model;
    public Instances data;
    public int folds;
    public double avg_f1 = 0;

    public KCrossVal(WekaModel model, Instances data, int folds){
        this.model = model;
        this.data = data;
        this.folds = folds;
    }

    public void execute() throws Exception {
        if (data.classAttribute().isNominal()){
            data.stratify(folds);
        }
        Evaluation overallEval = new Evaluation(data);
        for (int i = 0; i < folds; i++){
            Evaluation eval = new Evaluation(data);
            Instances train = data.trainCV(folds, i);
            Instances test = data.testCV(folds, i);
            this.model.buildClassifier(train);
            eval.evaluateModel(model.classifier, test);
            overallEval.evaluateModel(model.classifier, test);
            // this.avg_f1 += eval.fMeasure(1);
        }
        // this.avg_f1 = this.avg_f1/folds;
        // System.out.println("Average F1 score: " + avg_f1);
        System.out.println(overallEval.toSummaryString("10-fold cross validation evaluation results:\n", false));
        System.out.println(overallEval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(overallEval.toMatrixString("===Overall Confusion Matrix===\n"));

    }

    public void saveModel(String filePath) throws Exception {
        weka.core.SerializationHelper.write(filePath, this.model);
    }
}
