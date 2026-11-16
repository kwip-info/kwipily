import Foundation
import KwipMLCore

/// Current runnable-example delivery reuses bundle-authored teaching and runtime factories.
@MainActor enum BundledExampleLessonCopy {
    private static let reviewedIDs=TensorOperationAuthoring.componentIDs.union(Set(LearnCatalogContent.componentExampleEntries.filter {$0.item.category=="Layers & activations"}.map(\.componentID)))
    static func isReviewed(_ id:String)->Bool {reviewedIDs.contains(id) || LearnCatalogContent.componentExampleEntries.contains {$0.componentID==id && $0.item.category=="Training & evaluation"}}
    private static func layerPurpose(_ id:String)->String? {
        if id=="rotary_position" {return "An attention model needs to distinguish positions as well as token content. Rotary position rotates feature pairs in Query and Key so their relative positions can affect comparisons. It leaves Value on its separate path."}
        if id=="rms_norm" {return "Keep a feature group's magnitude on a useful scale without subtracting its mean. RMS normalization differs from Layer normalization: positive constant inputs need not become zero."}
        if ConvolutionConfiguration.componentIDs.contains(id) {return id.contains("transpose") ? "Spread shared learned windows into an output and add overlapping contributions. Transposed convolution controls geometry; it does not generally recover an arbitrary original image.":"Recognize a local pattern using the same learned window at multiple positions. Channels are measurement fields; the example constructs spatial features from synthetic table data, rather than importing photographs."}
        if PoolingConfiguration.componentIDs.contains(id) {return "Summarize nearby positions separately within each channel. Max chooses winners; Average combines a region using a sum and divisor. Compare the selected window or adaptive-output method with the actual captured values."}
        if NormalizationConfiguration.componentIDs.contains(id) {return "Bring a chosen group of model measurements onto a consistent scale. Decide which values share statistics and how training/evaluation use them; this is distinct from Preprocess fitting a Training table."}
        if RecurrentCellConfiguration.componentIDs.contains(id) {return "Define one update from a measurement and an explicit starting state. The cell's hidden state is data passed to a later step, not the weights learned during training; the cell never silently remembers a previous call."}
        if RecurrentSequenceConfiguration.componentIDs.contains(id) {return "Read ordered positions while updating a hidden summary. Compare the output at every position with the final hidden state; LSTM also carries separate cell memory."}
        if ["multihead_attention","scaled_dot_product_attention","transformer","transformer_encoder","transformer_decoder"].contains(id) {return "Let one sequence position use information from other allowed positions. Queries compare with keys, values contribute information, and masks restrict those comparisons. A full Transformer combines several such operations and feature transformations."}
        if ["bag_input","embedding_bag"].contains(id) {return "Represent a group of categories belonging to one observation. Embedding bag selects their shared lookup vectors and combines them into one result; item weights and packed boundaries remain explicit data."}
        if ["embedding","index_input"].contains(id) {return "Use a category's whole-number ID to select a learned vector. The numeric ID is a row label, not a continuous measurement or a similarity score."}
        if LinearConfiguration.componentIDs.contains(id) {return "Mix each observation's final feature direction into a new feature list or class scores. Weights control each contribution, bias supplies an offset, and training adjusts those parameters."}
        if ActivationConfiguration.componentIDs.contains(id) {return "Choose how feature values pass through the model. Most activations transform individual values; Softmax and LogSoftmax compare groups of scores. The choice changes the calculation, not the observation identities."}
        return nil
    }
    static func steps(_ id:String,learning:any BundledComponentLearningContent)->[LessonContentStep]? {
        guard isReviewed(id) else {return nil}
        let purposes:[String:String]=[
            "concat":"Two model paths can describe the same observation. Join their features along an existing direction while keeping the sources in a deliberate order.",
            "add":"Two paths can contribute corrections to the same feature positions. Add combines their values, with an optional fixed multiplier for the second source.",
            "stack":"Keep two equally shaped sources distinguishable by placing them along a new source direction instead of adding their values.",
            "permute":"A later layer may need the same numeric array with directions in a different order. Reordering coordinates is different from regrouping reading order with Reshape.",
            "transpose":"Swap the roles of two directions, such as rows and columns, while keeping each value attached to its coordinates.",
            "split":"A feature array may contain several consecutive groups. Split tensor exposes their named parts so later connections can use the intended group.",
            "chunk":"Request groups of positions, then inspect the parts that actually exist. A requested count can exceed the actual number of nonempty chunks.",
            "gather":"A second tensor of positions can select features from a source tensor. Those whole-number positions are data and may deliberately repeat.",
            "einsum":"Describe which numeric directions combine, remain or disappear. Named arrangements make common combinations available without writing an equation.",
            "unfold":"Sliding-window operations reuse nearby cells. Unfold exposes those small rectangles as columns so you can inspect what each window reads.",
            "fold":"Patch contributions may overlap when placed on a grid. Fold adds them at matching positions and leaves uncovered positions zero.",
            "reshape":"A layer may need a different feature layout. Reshape changes how the same reading order is grouped without creating new values.",
            "flatten":"A later layer may need a single feature direction. Flatten joins selected directions while keeping observations separate.",
            "matmul":"Combine rows from one table with columns from another by multiplying paired values and adding them. This is a common feature-mixing calculation."
        ]
        let layoutNote=["unfold","fold"].contains(id) ? " The model constructs small numeric grids from the example features; these are not imported photographs." : ""
        var intro=(purposes[id] ?? layerPurpose(id) ?? learning.subtitle)+"\n\nThis Example initially opens a lesson canvas. Expand Exercise choices & reference and use “"+learning.buttonTitle+"” to load its supplied editable workflow. Save existing work before replacing it.\n\nFollow the workflow:\n- Data: Table Input and preparation tools produce Training and Validation tables.\n- Model: input tensors pass through the featured operation and later layers to Model definition.\n- Train: maps prepared columns into the model, learns from Training and measures Validation.\n\nA tensor is an array of numbers. A sample is one observation; a batch contains several observations. Shape lists the size of each direction. N, or axis 0, counts observations and stays separate from sample directions."+layoutNote
        if id=="rotary_position" {
            intro=(layerPurpose(id) ?? learning.subtitle)+"\n\nA fresh Example opens a lesson canvas. Use “"+learning.buttonTitle+"” in Exercise choices & reference to open the supplied causal token model. It reads short synthetic text, encodes token IDs and creates aligned next-token windows. The model projects Query, Key and Value; only Query and Key pass through Rotary position before attention. Train learns from Training and measures Validation; Generate text uses the completed model and saved tokenizer.\n\nPositions, heads and feature-pair directions describe different things. A head is one parallel set of attention comparisons; token positions are places in the sequence. Shape [B, H, T, D] means observations, heads, token positions and features. The last feature width must support the configured pairs. RoPE itself does not mask future positions; the attention mask remains explicit. This tiny model is not a full Llama or a language-quality benchmark."
        }
        if ["bag_input","embedding_bag"].contains(id) {intro += "\n\nPacked item lists and offsets are special roles: their first direction counts items or boundaries, not independent observations. Use the configured bag layout rather than treating every axis as an ordinary batch axis."}
        var result=[LessonContent.step("goal","Meet the example workflow",intro),LessonContent.step("build","Open and trace the example","1. Open the supplied example using its button below.\n2. Find the featured operation in the model path and follow its incoming connections.\n3. Select it to inspect Configuration and its declared input shapes.\n4. Predict one value's destination before running.\n\nPlanned shapes describe the configured graph. They are not captured execution values.",kind:.canvasAction)]
        let headings:[String:[String]]=[
            "stack":["Keep the sources separate","Choose the new direction","Check matching inputs","Apply and compare"],
            "permute":["Reorder directions","Use every direction once","Keep existing projects readable"],
            "split":["Choose part sizes","Check actual outputs","Compare chunk requests"],
            "chunk":["Request a count","Connect actual parts","Use the consolidated tool"],
            "gather":["Connect values and positions","Check valid positions","Separate declared and measured values"],
            "einsum":["Choose an arrangement","Decide which operands enter","Check supported notation"],
            "unfold":["Configure window geometry","Inspect patch packing","Compare with Fold"],
            "fold":["Configure the output grid","Add overlapping contributions","Inspect captured values"]]
        let familyHeadings:[String]?
        if RecurrentAuthoringConfiguration.componentIDs.contains(id) {familyHeadings=["Understand the sequence","Use hidden memory","Read the output roles","Provide starting state","Interpret example evidence"]}
        else if ConvolutionConfiguration.componentIDs.contains(id) {familyHeadings=["Understand channels","Configure the window geometry","Compare captured output","Check groups and parameters"]}
        else if ["embedding_bag","bag_input"].contains(id) {familyHeadings=["Understand groups and offsets","Choose the combination","Separate IDs from influence","Handle padding and norm","Use the current tool"]}
        else if id=="rms_norm" {familyHeadings=["Choose the normalization group","Compare centering behavior","Configure learned scale","Run and inspect"]}
        else if id=="rotary_position" {familyHeadings=["Follow Query and Key","Choose the position direction","Keep a pairing convention","Train and inspect the model","Keep position semantics consistent"]}
        else {familyHeadings=nil}
        result += learning.paragraphs.enumerated().map {index,paragraph in
            var lines=paragraph.components(separatedBy:"\n")
            let first=lines.first ?? ""
            let title:String
            if first.hasPrefix("**"),first.hasSuffix("**") {title=String(first.dropFirst(2).dropLast(2));lines.removeFirst()}
            else {title=(headings[id] ?? familyHeadings).flatMap {$0.indices.contains(index) ? $0[index]:nil} ?? learning.disclosureTitle+" · \(index+1)"}
            return LessonContent.step("explain.\(index)",title,lines.joined(separator:"\n"))
        }
        result.append(.init(id:"inspect",title:"Run and compare the evidence",blocks:[
            .init(id:"inspect.body",kind:.canvasAction,title:"Try it on the canvas",text:"1. Run Flow using the supplied settings.\n2. Select the featured operation and inspect its captured Output.\n3. Compare input and output shapes, then trace a few actual values.\n4. Change one setting, Apply and rerun before interpreting the new output.\n\nTrain's predictions table is a separate result. A correctly shaped tensor does not establish prediction quality."),
            .init(id:"inspect.formula",kind:.visual,title:"Optional relationship",text:learning.formula,visual:.init(kind:.equation)),
            .init(id:"inspect.question",kind:.question,title:"Check your understanding",text:"What changed: values, positions, or the number of directions? Which observations stayed independent? Explain the difference between a planned shape and a captured value.\n\nThis runnable example is unscored. A separate Construction assessment uses measured criteria and keeps its private fixture tables hidden; opening or running this workflow does not award that assessment's receipt.")]))
        return result
    }
}
