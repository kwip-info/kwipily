import Foundation

public enum EmbeddingLearningTask:String,CaseIterable,Codable {
    case gather = "embedding-gather"
    case paddingFrequency = "embedding-padding-frequency"
    case renormalize = "embedding-renormalize"

    public var title:String {
        switch self {
        case .gather:return "Embedding · gather and accumulate"
        case .paddingFrequency:return "Embedding · padding and frequency"
        case .renormalize:return "Embedding · forward weight changes"
        }
    }
    public var configuration:EmbeddingConfiguration {
        get throws {
            try EmbeddingConfiguration(values:["num_embeddings":4,"embedding_dim":3,
                "padding_idx":self == .paddingFrequency ? 0 : .null,
                "max_norm":self == .renormalize ? 1 : .null,"norm_type":2,
                "scale_grad_by_freq":.bool(self == .paddingFrequency),"sparse":false,"freeze":false])
        }
    }
    public var task:String {
        let base="An Embedding is a shared lookup table: a whole-number category index selects one row, called a vector, containing three numbers. Add Input, Embedding and Model Output; connect them in that order. The lesson supplies a table with four rows. Set Table rows to 4 and Vector dimensions to 3. Keep Sparse gradients and Freeze learned weights disabled for these graded tasks. The output keeps every selector position and adds three selected values at each position; repeated selectors repeat a row rather than adding rows together. Choose Check construction to inspect the selected values, the table after lookup and its gradient. A gradient measures how small changes to the table values affect the result; KwipML checks it automatically. Category indices are discrete row labels, so there is no input gradient for them. No code or hand-entered gradient answers are required."
        switch self {
        case .gather:return base+" Leave Padding row and Maximum norm unset; disable Scale gradients by frequency. Repeated uses add their influence to the same table row. Unselected rows receive no contribution, and lookup leaves the table unchanged."
        case .paddingFrequency:return base+" Enable Use padding row and set Padding row to 0 and enable Scale gradients by frequency; leave Limit vector norm disabled. Padding is a reserved row: its supplied values remain visible in the output, but it receives no gradient contribution. For other rows, frequency means the number of selections across the entire input; divide the total influence by that count. The lookup leaves the table unchanged."
        case .renormalize:return base+" Leave Use padding row disabled, disable Scale gradients by frequency, and enable Limit vector norm and set Maximum norm to 1 with Norm order 2. Norm means vector length; order 2 measures length from the sum of squared values. Any selected row longer than 1 is shortened before lookup; unselected rows stay unchanged. This changes the table even with Freeze enabled in a general model, though these graded tasks require Freeze disabled. The automatic gradient check treats that shortening as a separate forward change."
        }
    }
}

/// Hidden fixtures use small integer coordinates represented exactly in the shared
/// numeric fixture envelope. The worker adapter must validate then cast to int64;
/// this envelope does not authorize conversion of arbitrary floating-point inputs.
public enum EmbeddingAssessment {
    public static func plan(task:EmbeddingLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func next()->Double {
            state=state &* 6364136223846793005 &+ 1442695040888963407
            return Double((state >> 32)%97)/8-6
        }
        let configuration=try task.configuration
        let fixtures=try (0..<4).map { index -> LearningFixture in
            let shape=index%2==0 ? [1,4] : [2,2,2]
            let indices:[Int64]=index%2==0 ? [0,2,2,1] : [2,0,1,2,1,0,2,1]
            let table=LearningTensor(shape:[4,3],values:index==0 ? [3,4,0,1,-2,2,6,8,0,9,12,0] : (0..<12).map {_ in next()})
            let upstream=LearningTensor(shape:shape+[3],values:(0..<(indices.count*3)).map {Double(($0*7+index)%13)-5})
            let reference=try EmbeddingLessonReference(indices:indices,shape:shape,table:table,configuration:configuration)
            guard let gradient=try reference.tableGradient(for:upstream) else {throw LearningContractError("Embedding assessment requires trainable table parameters.")}
            return LearningFixture(id:"\(task.rawValue)-\(seed)-\(index)",inputs:["indices":.init(shape:shape,values:indices.map(Double.init)),"table":table,"upstream":upstream],expected:["output":reference.output,"tableAfterForward":reference.tableAfterForward,"tableGradient":gradient])
        }
        func criterion(_ key:String,_ description:String,_ hint:String)->LearningCriterion {
            .init(id:key,description:description,nextStep:hint,rule:.tensor(key:key,absoluteTolerance:1e-6,relativeTolerance:1e-6))
        }
        let criteria=[
            criterion("output","Select complete rows in the original selector order and keep every grouping direction.","For each index position, copy the selected row's three coordinates. Repetition repeats a row; it does not add rows together. Apply any forward norm limit first."),
            criterion("tableAfterForward","Inspect the table after lookup, including changes to selected rows when a length limit is enabled.","Only selected rows may be renormalized. Padding suppression is a gradient policy, not an instruction to replace supplied padding values with zeros."),
            criterion("tableGradient","Measure the influence on each shared row using the requested padding and frequency settings.","Start with zeros shaped like W. Scatter-add each upstream row at its selected index. Suppress padding and divide by occurrence count only when requested. Do not differentiate integer indices or the no-grad renormalization.")
        ]
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"lookup."+task.rawValue,task:task.task,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:["embedding":ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate();return plan
    }
}
