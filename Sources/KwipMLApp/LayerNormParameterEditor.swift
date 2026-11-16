import SwiftUI
import KwipMLCore

/// Paged explicit parameter entry, sharing the canvas's persisted drafts and undo history.
struct LayerNormParameterEditor:View {
    @ObservedObject var canvas:TensorLearningCanvasAdapter
    let node:GraphNode
    @State private var pages:[String:Int]=[:]
    private var active:Bool {canvas.historyContextIsActive && canvas.selectedNodeIDs == [node.id]}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Divider()
            SectionLabel(title:node.type == "prelu" ? "Learned negative multipliers":"Normalization parameters")
            if node.type == "prelu",let config=try? RemainingActivationConfiguration(componentID:node.type,values:node.config),let count=config.numParameters {
                Text("A slope multiplies negative measurements. One slope is shared by all channels; otherwise each channel has its own. Initial slope sets starting values; the values below are the actual learned parameters used in this lesson.").font(.caption).foregroundStyle(MLTheme.muted)
                Button("Initialize / reset slopes from Initial slope") {
                    guard active else {return};_ = canvas.initializeNormalizationParameters(nodeID:node.id)
                }.buttonStyle(FlatButtonStyle()).accessibilityLabel("Initialize or reset learned slopes")
                if let tensor=canvas.session.draft?.parameters?[node.id]?["weight"],tensor.shape == [count] {parameter("weight",tensor:tensor)}
                else {Text("Initialize the slopes to create values matching Slope count. Undo restores the previous configuration and values.").font(.caption).foregroundStyle(MLTheme.rose)}
                Toggle("Inspect this block’s learned-slope influence",isOn:Binding(get:{canvas.session.draft?.gradientNodeID == node.id},set:{selected in
                    guard active else {return};_ = canvas.setGradientTarget(selected ? node.id:nil)
                })).toggleStyle(.checkbox).font(.caption)
                Text("The checker measures how each learned multiplier affects the result, adding its contributions wherever it is reused.").font(.caption).foregroundStyle(MLTheme.muted)
            } else if StatefulNormalizationConfiguration.componentIDs.contains(node.type), let config=try? StatefulNormalizationConfiguration(componentID:node.type,values:node.config) {
                Text("Channels share one scale and optional shift across their positions. Running statistics, when enabled, belong to the checker and are not editable answers.").font(.caption).foregroundStyle(MLTheme.muted)
                if config.affine {
                    Button("Initialize / reset scale to 1 and shift to 0") {guard active else {return};_ = canvas.initializeNormalizationParameters(nodeID:node.id)}.buttonStyle(FlatButtonStyle()).accessibilityLabel("Initialize or reset normalization parameters")
                    ForEach(config.bias ? ["weight","bias"]:["weight"],id:\.self) {name in
                        if let tensor=canvas.session.draft?.parameters?[node.id]?[name],tensor.shape == [config.channelCount] {parameter(name,tensor:tensor)}
                        else {Text("Initialize values matching Channels; Undo restores previous values.").font(.caption).foregroundStyle(MLTheme.rose)}
                    }
                    Toggle("Inspect this block’s learned-adjustment influence",isOn:Binding(get:{canvas.session.draft?.gradientNodeID == node.id},set:{selected in guard active else {return};_ = canvas.setGradientTarget(selected ? node.id:nil)})).toggleStyle(.checkbox).font(.caption)
                } else {Text("Learned adjustments are off. This block owns no scale or shift values.").font(.caption).foregroundStyle(MLTheme.muted)}
            } else if let config=try? LayerNormConfiguration(values:node.config) {
                Text("Trailing dimensions are normalized together within each observation. Population variance and current-input statistics are used in training and evaluation. Epsilon is added inside the square root.").font(.caption).foregroundStyle(MLTheme.muted)
                if config.elementwiseAffine {
                    Text("Enter one scale and optional bias per trailing coordinate. These values are shared across preceding sample axes and observations.").font(.caption)
                    Button("Initialize / reset scale to 1 and bias to 0") {
                        guard active else {return}
                        _ = canvas.initializeNormalizationParameters(nodeID:node.id)
                    }.buttonStyle(FlatButtonStyle()).accessibilityLabel("Initialize or reset normalization parameters")
                    ForEach(config.bias ? ["weight","bias"] : ["weight"],id:\.self) {name in
                        if let tensor=canvas.session.draft?.parameters?[node.id]?[name] {
                            if tensor.shape == config.normalizedShape {parameter(name,tensor:tensor)}
                            else {Text("\(name == "weight" ? "Scale":"Bias") shape no longer matches the trailing dimensions. Initialize/reset to create matching values; Undo restores the previous state.").font(.caption).foregroundStyle(MLTheme.rose)}
                        } else {Text("\(name == "weight" ? "Scale":"Bias") values have not been initialized.").font(.caption).foregroundStyle(MLTheme.rose)}
                    }
                    Toggle("Inspect this component’s parameter gradients",isOn:Binding(get:{canvas.session.draft?.gradientNodeID == node.id},set:{selected in
                        guard active else {return};_ = canvas.setGradientTarget(selected ? node.id:nil)
                    })).toggleStyle(.checkbox).font(.caption)
                    Text("Only one component is selected for parameter-gradient inspection. Input gradients still cover the whole construction.").font(.caption).foregroundStyle(MLTheme.muted)
                } else {
                    Text("Affine scaling is disabled, so this component owns no scale or bias tensors. The bias option has no effect until affine scaling is enabled.").font(.caption).foregroundStyle(MLTheme.muted)
                }
            }
        }.disabled(!active)
    }
    private func coordinates(_ index:Int,shape:[Int])->String {
        var remainder=index,result:[Int]=[]
        for dimension in shape.reversed() {result.insert(remainder % dimension,at:0);remainder /= dimension}
        return "["+result.map(String.init).joined(separator:", ")+"]"
    }
    private func parameter(_ name:String,tensor:LearningTensor)->some View {
        let title=node.type == "prelu" ? "Learned slope":name == "weight" ? "Scale γ":"Bias β"
        let page=min(max(pages[name] ?? 0,0),max(0,(tensor.values.count-1)/8))
        let start=page*8,end=min(start+8,tensor.values.count)
        return VStack(alignment:.leading,spacing:6) {
            Text("\(title) · shape [\(tensor.shape.map(String.init).joined(separator:", "))]").font(.caption.weight(.semibold))
            ForEach(start..<end,id:\.self) {index in
                VStack(alignment:.leading,spacing:3) {
                    Text(title+" "+coordinates(index,shape:tensor.shape)).font(.system(size:10,design:.monospaced))
                    TextField(title,text:canvas.parameterTextBinding(nodeID:node.id,name:name,index:index)).flatField().accessibilityLabel(title+" "+coordinates(index,shape:tensor.shape))
                        .onSubmit {guard active else {return};_ = canvas.applyPendingParameter(nodeID:node.id,name:name,index:index)}
                    if canvas.pendingParameter(nodeID:node.id,name:name,index:index) != nil {
                        HStack {
                            Button("Apply") {guard active else {return};_ = canvas.applyPendingParameter(nodeID:node.id,name:name,index:index)}
                            Button("Discard") {guard active else {return};canvas.discardPendingParameter(nodeID:node.id,name:name,index:index)}
                        }.buttonStyle(.borderless).font(.caption)
                    }
                }.id(canvas.historyContextID+node.id+name+String(index))
            }
            if tensor.values.count>8 {
                HStack {
                    Button("Previous") {pages[name]=page-1}.disabled(start == 0)
                    Text("\(start+1)–\(end) of \(tensor.values.count)").font(.caption)
                    Button("Next") {pages[name]=page+1}.disabled(end == tensor.values.count)
                }.buttonStyle(.borderless)
            }
        }
    }
}
