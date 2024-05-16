package Evaluation;

import Model.WekaModel;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class Evaluate {
    public WekaModel model;
    public Instances train;
    public Instances test;

    public Evaluate(WekaModel model, Instances train, Instances test) {
        this.model = model;
        this.train = train;
        this.test = test;
    }

    public void execute() throws Exception{
        this.model.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(model.classifier, test);
        System.out.println(eval.toSummaryString("Evaluation results:\n", false));
        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval.toMatrixString("===Overall Confusion Matrix===\n"));
    }
}
