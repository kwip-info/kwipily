import Foundation

/// Versioned teaching sequence over the same executable workflow used by authoring.
/// Prefix documents are new-course seeds, never replacements for a learner's edits.
public struct WorkflowGuidedBuildPlan {
    public struct Step:Equatable {
        public let address:WorkflowNodeAddress
        public let title:String
        public let explanation:String
    }
    public let id:String
    public let version:Int
    public let steps:[Step]
    public let completeDocument:WorkflowDocument

    /// Append one stage through the ordinary atomic command/history path. Existing
    /// learner nodes, configuration and wires are never replaced by a template.
    public func insertion(at stage:Int)throws->WorkflowCommand {
        guard steps.indices.contains(stage),
              let node=completeDocument.nodes.first(where:{$0.address == steps[stage].address}) else {
            throw DataFlowContractError("Choose a valid guided-build stage.")
        }
        return .insertConnected(node,incoming:completeDocument.edges.filter {$0.to.address == node.address})
    }

    public func document(completedSteps:Int)throws->WorkflowDocument {
        guard (0...steps.count).contains(completedSteps) else {throw DataFlowContractError("Choose a valid guided-build stage.")}
        let included=Set(steps.prefix(completedSteps).map(\.address))
        var result=completeDocument
        result.guidedBuild = .init(courseID:id,version:version,completedSteps:completedSteps)
        result.nodes=result.nodes.filter {included.contains($0.address)}
        result.edges=result.edges.filter {included.contains($0.from.address) && included.contains($0.to.address)}
        let owners=Set(result.nodes.filter {WorkflowModelBoundary.isDefinition($0)}.map { $0.address.localID })
        result.modelInputs=result.modelInputs.map {$0.filter {owners.contains($0.key)}}
        result.modelSourceVersions=result.modelSourceVersions.map {$0.filter {owners.contains($0.key)}}
        try WorkflowCodec.validate(result)
        return result
    }

    public static func parallelClassifier(projectID:String = UUID().uuidString,version:Int=5)throws->Self {
        var document=try WorkflowTrainRunSetup.migrate(WorkflowExamples.preparedClassifier(projectID:projectID,device:"auto",model:ExampleFlows.parallelClassifier))
        document.name="Learn · Build a complete classification workflow"
        func step(_ domain:WorkflowDomain,_ id:String,_ title:String,_ text:String)->Step {
            .init(address:.init(domain:domain,localID:id),title:title,explanation:text)
        }
        var steps:[Step]=[
            step(.data,"observations","Begin with observations","A table contains observations in rows and variables in columns. Inspect the CSV source before choosing which columns become features or targets."),
            step(.data,"cast","Choose types with Columns","Use the Columns grid to convert the two features to decimals and the class target to an integer. Storage type and modeling role are different decisions."),
            step(.data,"split","Reserve validation and final test","Partition reserves one quarter for validation and one quarter for final test; training receives the remaining half. Fit preprocessing and model parameters using training observations only; do not tune on a final test set."),
            step(.data,"fit","Prepare all populations with Preprocess","Choose Standard scaling for each feature in Preprocess. It learns a mean μ and scale σ using training rows only, then prepares training, validation and test. Standardization applies z = (x − μ) / σ. Learning these statistics from validation would leak information."),
            step(.data,"training","Inspect prepared training rows","View receives Preprocess’s training output. This changes feature coordinates, not the target. Inspect the captured table after running."),
            step(.data,"validation","Inspect prepared validation rows","View receives Preprocess’s validation output, prepared with training μ and σ. Fitting a second scaler would give the partitions different coordinate systems."),
            step(.model,"input","Declare the model input settings","This stage prepares the internal input boundary automatically. Each batch is N × 2; N may vary. Edit the feature count in Model definition after its architecture is connected."),
            step(.model,"branch-a","Learn a first projection","Linear computes Y = XWᵀ + b. W has shape 8 × 2; an N × 2 batch becomes N × 8. Each output is a learned weighted sum."),
            step(.model,"activation-a","Introduce nonlinearity","ReLU maps each value x to max(0,x). Without nonlinearities, a chain of affine layers still represents one affine map."),
            step(.model,"branch-b","Learn an independent branch","A second 8 × 2 weight matrix reads the same features. Parallel paths share input observations, not parameters."),
            step(.model,"activation-b","Give the second branch a different response","Tanh maps values into (−1,1). Its derivative 1 − tanh²(x) becomes small in saturated regions; bounded output does not guarantee easy optimization."),
            step(.model,"merge","Preserve both feature sets","Concatenate joins the two N × 8 matrices along the feature axis to make N × 16. Addition would require matching shapes and retain only eight features."),
            step(.model,"classifier","Produce class scores","A 2 × 16 weight matrix projects the merged representation to two logits per observation. These are unnormalized scores, not probabilities."),
            step(.model,"output","Expose the prediction tensor","The output identifies the N × 2 logits consumed by the classification objective. Cross-entropy handles log-softmax internally; inserting softmax here would change the intended objective."),
            step(.control,"model","Configure the model and its input","Select Model definition to inspect Floating values, two features and the two branch entry ports. Input is model configuration; no separate Input block is needed. The batch dimension is supplied during execution."),
            step(.control,"mapping","Map features and target explicitly","Connect Preprocess’s Training and Validation outputs directly to Training data; View remains an inspection sink. Choose signal then context as model inputs and label as the target. Both partitions use the same mapping; the target must not become an input feature."),
            step(.control,"train","Execute and inspect evidence","Train connects the model and mapped partitions. Its Run settings own device, epoch budget, batches and seed; Partition owns its population seed. Change one factor at a time. Run flow, inspect bottom-dock tables and messages, then verify loss and held-out accuracy in History. This guided build is practice, not a mastery assessment.")
        ]
        guard [3,4,5].contains(version) else {throw DataFlowContractError("Unsupported guided classification version.")}
        if version>=4 {
            document=try WorkflowModelBoundary.consolidate(document)
            steps.removeAll {$0.address == .init(domain:.control,localID:"model")}
            if let i=steps.firstIndex(where:{$0.address == .init(domain:.model,localID:"output")}) {
                steps[i]=step(.model,"output","Define the model boundary","Connect the final N × 2 logits to Model definition, then connect its Model output directly to Train. Its Configuration owns input features and entry ports. Cross-entropy consumes logits; no extra softmax or pass-through model bridge is needed.")
            }
        }
        if version>=5 {
            document=try WorkflowTrainMapping.migrate(document)
            steps.removeAll {$0.address == .init(domain:.control,localID:"mapping")}
            if let i=steps.firstIndex(where:{$0.address == WorkflowExamples.trainAddress}) {
                steps[i]=step(.control,"train","Map columns and train","Connect Model definition, Preprocess Training and Validation directly to Train. In Data mapping, choose signal then context as inputs and label as target. Run settings control device, epochs and seed. Both partitions use the same ordered mapping; never use the target as an input feature.")
            }
        }
        guard Set(steps.map(\.address)) == Set(document.nodes.map(\.address)),steps.count == document.nodes.count else {
            throw DataFlowContractError("The guided build no longer matches its workflow. Update the lesson version and stages.")
        }
        return .init(id:"kwip.course.universal-classifier",version:version,steps:steps,completeDocument:document)
    }
}
