package Evaluation;

import weka.core.Instances;

import weka.associations.Apriori;
import weka.associations.AssociationRule;
import weka.associations.AssociationRules;

import java.util.List;

public class EvaluateApriori {
    private Instances data;

    public EvaluateApriori(Instances data) {
        this.data = data;
    }

    // Overloaded method to evaluate Apriori model
    public void execute() throws Exception {
        // Initialize and build Apriori model
        Apriori apriori = new Apriori();
        apriori.buildAssociations(data);

        // Get the generated association rules
        AssociationRules rules = apriori.getAssociationRules();
        List<AssociationRule> ruleList = rules.getRules();

        // Evaluate the rules (basic evaluation by printing)
        System.out.println("\nEvaluation of Generated Rules:");
        for (AssociationRule rule : ruleList) {
            System.out.println("Rule: " + rule);
            System.out.println("Support: " + rule.getPrimaryMetricValue()); // Usually the support
        }

    }
}




