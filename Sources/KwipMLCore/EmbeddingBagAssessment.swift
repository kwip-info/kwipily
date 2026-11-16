import Foundation

public enum EmbeddingBagLearningTask:String,CaseIterable,Codable {
    case fixedSum="bag-fixed-sum",packedMean="bag-packed-mean",packedMax="bag-packed-max"
    case fixedWeightedSum="bag-fixed-weighted-sum",packedWeightedSum="bag-packed-weighted-sum",paddingNorm="bag-padding-norm"
    public var title:String {
        switch self {
        case .fixedSum:return "Combine a fixed group"
        case .packedMean:return "Average groups of different sizes"
        case .packedMax:return "Keep the strongest values"
        case .fixedWeightedSum:return "Give items different influence"
        case .packedWeightedSum:return "Weight groups of different sizes"
        case .paddingNorm:return "Ignore padding and limit vector length"
        }
    }
    public var configuration:EmbeddingBagConfiguration {get throws {
        var values=EmbeddingBagConfiguration.defaults
        values["num_embeddings"]=4;values["embedding_dim"]=3
        values["layout"] = .string(self == .fixedSum || self == .fixedWeightedSum ? "fixed":"packed")
        values["mode"] = .string(self == .packedMean ? "mean":self == .packedMax ? "max":"sum")
        values["per_sample_weights"] = .bool(self == .fixedWeightedSum || self == .packedWeightedSum)
        values["include_last_offset"] = .bool(self == .packedWeightedSum)
        if self == .paddingNorm {values["padding_idx"]=0;values["max_norm"]=1}
        return try EmbeddingBagConfiguration(values:values)
    }}
    public var sourceConfiguration:BagSourceConfiguration {get throws {
        let c=try configuration
        return try BagSourceConfiguration(values:["layout":.string(c.layout),"maximum_items_per_bag":3,"per_sample_weights":.bool(c.usesPerSampleWeights),"include_last_offset":.bool(c.includeLastOffset)])
    }}
    public var task:String {
        let base="A bag is one observation's group of category IDs. Each ID is a whole number selecting a row in a shared table. A selected row is a vector: an ordered list of numbers. Place Bag Input, Embedding bag and Model Output. Connect every enabled source connector to the matching connector on Embedding bag, then connect its output to Model Output. The lesson supplies the source rows and table; no code or hand-entered answers are needed. Set Table rows to 4, Vector dimensions to 3, and Maximum items per group to 3. Keep Freeze learned weights, Sparse gradients and Scale gradients by frequency disabled. Choose Check construction to see each group's combined vector, the table after the operation and the measured influence on table values. A gradient measures how a small change affects the result. An offset marks where a group starts in a packed list. Whole-number IDs and offsets have no gradient available. "
        switch self {
        case .fixedSum:return base+"Choose Fixed rows in Group layout on both components and Sum in Combine selected vectors. Disable item weights and Include final boundary. Leave Use padding row and Limit vector norm disabled. Add every selected vector within each group; repeated IDs contribute again. Keep the groups separate."
        case .packedMean:return base+"Choose Packed groups in Group layout on both components and Mean in Combine selected vectors. Disable item weights and Include final boundary. Leave Use padding row and Limit vector norm disabled. Source lengths tell how many cells belong to each group. Average each group's actual items. Empty groups keep their output row and return zeros. Offsets mark group boundaries, not table rows."
        case .packedMax:return base+"Choose Packed groups in Group layout on both components and Maximum in Combine selected vectors. Disable item weights and Include final boundary. Leave Use padding row and Limit vector norm disabled. Keep the largest number separately at each vector position, rather than selecting one whole vector. Empty groups keep their output row and return zeros."
        case .fixedWeightedSum:return base+"Choose Fixed rows in Group layout on both components, enable item weights and choose Sum. Disable Include final boundary. Leave Use padding row and Limit vector norm disabled. A weight controls an item's influence: multiply its selected vector by that weight before adding within its group. Connect the real weight source. KwipML also checks influence on those weights."
        case .packedWeightedSum:return base+"Choose Packed groups in Group layout on both components, enable item weights and Include final boundary, and choose Sum. Leave Use padding row and Limit vector norm disabled. Each weight travels with its item. The final offset marks the end of all items. Keep empty groups and their observation positions; they return zeros. KwipML also checks influence on the original source weights, including unused cells."
        case .paddingNorm:return base+"Choose Packed groups in Group layout on both components and Sum in Combine selected vectors. Disable item weights and Include final boundary. Enable Use padding row with Padding row 0, then Limit vector norm with Maximum norm 1 and Norm order 2. Padding is a reserved row excluded from the combination and from table influence. Norm means vector length; order 2 measures the usual straight-line distance from zero using squared coordinates. Selected rows longer than 1 are shortened before combining; unused rows stay unchanged. This forward change is separate from learned gradient updates."
        }
    }
}

public enum EmbeddingBagAssessment {
    public static func plan(task:EmbeddingBagLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func next()->Double {state=state &* 6364136223846793005 &+ 1442695040888963407;return Double((state>>32)%97)/8-6}
        let configuration=try task.configuration
        let lengthVariants=[[3,0,1],[0,2,3],[1,3,0],[0,0,0]]
        let fixtures=try (0..<4).map {index->LearningFixture in
            let shift=Int((seed>>8)%4)
            let rowIDs=[1,2,1,0,3,2,3,1,0].map {Double(($0+shift+index)%4)}
            let rows=LearningTensor(shape:[3,3],values:rowIDs)
            let lengths=configuration.layout == "packed" ? lengthVariants[index]:[3,3,3]
            let table=LearningTensor(shape:[4,3],values:(0..<12).map {_ in next()})
            let weights=configuration.usesPerSampleWeights ? LearningTensor(shape:[3,3],values:(0..<9).map {_ in next()/3}):nil
            let upstream=LearningTensor(shape:[3,3],values:(0..<9).map {_ in next()})
            let reference=try EmbeddingBagLessonReference(sourceRows:rows,lengths:lengths,table:table,itemWeights:weights,configuration:configuration)
            var inputs=["sourceRows":rows,"table":table,"upstream":upstream]
            if configuration.layout == "packed" {inputs["lengths"]=LearningTensor(shape:[3],values:lengths.map(Double.init))}
            if let weights {inputs["sourceWeights"]=weights}
            var expected=["output":reference.output,"tableAfterForward":reference.tableAfterForward]
            expected["tableGradient"]=try reference.tableGradient(for:upstream)
            if weights != nil {expected["sourceWeightsGradient"]=try reference.itemWeightGradient(for:upstream)}
            return LearningFixture(id:"\(task.rawValue)-\(seed)-\(index)",inputs:inputs,expected:expected)
        }
        let descriptions=[
            "output":("Combine each group's values and preserve every observation row.","Check the mode, repeated items and empty groups. Keep item weights paired with their category IDs."),
            "tableAfterForward":("Keep ordinary lookups unchanged and apply a requested length limit only to selected rows.","Check Padding row and Maximum norm. Padding is excluded from the combination; a length limit is a separate forward table change."),
            "tableGradient":("Measure influence on the shared table using the requested combination.","Repeated selections share a table row. Mean shares influence across actual items; max routes it through each position's first winner. Padding contributes no influence."),
            "sourceWeightsGradient":("Measure influence on each real source weight without mixing groups.","A weight's influence uses its selected vector. Unused source cells and padding have no contribution; keep each item with its own group.")]
        let keys=["output","tableAfterForward","tableGradient"]+(configuration.usesPerSampleWeights ? ["sourceWeightsGradient"]:[])
        let criteria=keys.map {key->LearningCriterion in
            let copy=descriptions[key]!
            return .init(id:key,description:copy.0,nextStep:copy.1,rule:.tensor(key:key,absoluteTolerance:3e-6,relativeTolerance:3e-5))
        }
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"bags."+task.rawValue,task:task.task,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:["bag_input":ContractVersion(),"embedding_bag":ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate();return plan
    }
}
