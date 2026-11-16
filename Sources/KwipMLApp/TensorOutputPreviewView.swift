import SwiftUI
import AppKit

/// Bounded presentation of captured evidence; no tensor queries or source reads.
struct TensorOutputPreviewView:View,Equatable {
    let preview:[String:Any]
    var selectedPort:String?=nil
    var family:ComponentVisualFamily = .tensorLayout
    var accent:Color?=nil
    var evidenceLabel:String?=nil
    var fillsViewport=false
    var renderIdentity:String?=nil
    static func ==(lhs:Self,rhs:Self)->Bool {
        guard let identity=lhs.renderIdentity else {return false}
        return identity==rhs.renderIdentity && lhs.selectedPort==rhs.selectedPort && lhs.family==rhs.family && lhs.fillsViewport==rhs.fillsViewport && lhs.evidenceLabel==rhs.evidenceLabel && lhs.accent==rhs.accent
    }
    @Environment(\.colorScheme) private var colorScheme
    @State private var activePort:String?
    private var outputs:[String:[String:Any]] {TensorPreviewFormatting.outputs(preview)}
    private var ports:[String] {
        if let selectedPort {return outputs[selectedPort] == nil ? []:[selectedPort]}
        let keys=outputs.keys.sorted()
        guard let choice=TensorPreviewFormatting.selectedOutputPort(available:keys,preferred:activePort) else {return []}
        return [choice]
    }
    private var minimumViewportHeight:CGFloat {
        guard let port=ports.first,let value=outputs[port] else {return 120}
        return (TensorSampleInspection(value).shape?.count ?? 0)>2 ? 300:190
    }
    var body:some View {
        if fillsViewport {
            GeometryReader {viewport in
                ThemedScrollView {
                    // Values can fill the dock, while heatmap controls and overview
                    // rows must contribute their full height to the scroll document.
                    content.frame(minHeight:max(minimumViewportHeight,viewport.size.height),alignment:.topLeading)
                }
            }
        } else {content}
    }
    private var content:some View {
        VStack(alignment:.leading,spacing:10) {
            if ports.isEmpty {Text("No captured output for this port.").font(.caption).foregroundStyle(MLTheme.muted)}
            ForEach(ports,id:\.self) {port in
                if let value=outputs[port] {
                    TensorCapturedPortView(port:port,sample:TensorSampleInspection(value),accent:accent ?? (colorScheme == .dark ? family.primary:family.secondary),evidenceLabel:evidenceLabel,portChoices:selectedPort == nil ? outputs.keys.sorted():[],onSelectPort:{activePort=$0})
                }
            }
        }
    }
}

private struct TensorCapturedPortView:View {
    let port:String
    let sample:TensorSampleInspection
    let accent:Color
    let evidenceLabel:String?
    let portChoices:[String]
    let onSelectPort:(String)->Void
    @State private var mode="matrix"
    @State private var heatmapMode=false
    @Environment(\.configureTensorCapture) private var configureCapture
    @State private var shading=false
    @State private var rowAxis:Int?
    @State private var columnAxis:Int?
    @State private var fixed:[Int:Int]=[:]
    @State private var fixedDrafts:[Int:String]=[:]
    @State private var indexError:String?
    @State private var selection:TensorCapturedGrid.Selection?
    @State private var showsEvidence=false
    private var shape:[Int] {sample.shape ?? []}
    private var matrixEligible:Bool {sample.shape != nil && shape.count>=2}
    private var row:Int {rowAxis ?? max(0,shape.count-2)}
    private var column:Int {columnAxis ?? max(0,shape.count-1)}
    private var fixedIndices:[Int:Int] {Dictionary(uniqueKeysWithValues:shape.indices.filter {$0 != row && $0 != column}.map {($0,fixed[$0] ?? 0)})}
    private var fixedIndexError:String? {
        for axis in fixedIndices.keys.sorted() {
            guard let text=fixedDrafts[axis] else {continue}
            guard let index=Int(text),shape[axis]>0,(0..<shape[axis]).contains(index) else {
                return "Axis \(axis) requires an index from 0 through \(max(0,shape[axis]-1))."
            }
        }
        return indexError
    }
    private var matrix:Result<TensorSampleInspection.Matrix,Error> {
        Result {
            let columns=min(256,max(1,shape.indices.contains(column) ? shape[column]:1))
            let rows=min(max(1,shape.indices.contains(row) ? shape[row]:1),256/columns)
            return try sample.matrix(rowAxis:row,columnAxis:column,fixedIndices:fixedIndices,maximumRows:rows,maximumColumns:columns)
        }
    }
    private var identity:String {
        sample.dtype+"|"+String(describing:sample.shape)+"|"+sample.availability+"|"+(evidenceLabel ?? "")+"|"+sample.rows.map {"\($0.id):\($0.value):\($0.state):\(String(describing:$0.coordinates))"}.joined(separator:";")
    }
    var body:some View {
        VStack(alignment:.leading,spacing:0) {
            ViewThatFits(in:.horizontal) {
                HStack(spacing:12) {metadata;Spacer(minLength:4);controls}
                VStack(alignment:.leading,spacing:8) {metadata;controls}
            }.padding(9).background(MLTheme.panel)
            Rectangle().fill(MLTheme.rule).frame(height:1)
            if heatmapMode {
                if let heatmap=sample.heatmap,heatmap.available || !(sample.heatmapPlayback?.frames.isEmpty ?? true) {TensorHeatmapPlaybackView(final:heatmap,playback:sample.heatmapPlayback,accent:accent)}
                else {VStack(alignment:.leading,spacing:8) {
                    Text(sample.heatmap?.reason ?? (sample.heatmap?.coverage=="empty" ? "Empty tensor":"No heatmap captured for this run")).font(.caption).foregroundStyle(MLTheme.muted)
                    if let configureCapture {Button("Configure next run",action:configureCapture).buttonStyle(FlatButtonStyle())}
                }.padding(12).frame(maxWidth:.infinity,maxHeight:.infinity,alignment:.topLeading)}
            } else if matrixEligible,mode == "matrix" {
                if shape.count>2 {sliceControls.padding(9).background(accent.opacity(0.04))}
                if let fixedIndexError {
                    unavailable(fixedIndexError)
                } else {
                    switch matrix {
                    case .success(let value):
                        grid(matrixGrid(value))
                    case .failure(let error):unavailable(error.localizedDescription)
                    }
                }
            } else {
                if sample.rows.isEmpty {unavailable(sample.availability)} else {grid(coordinateGrid)}
            }
        }.background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
            .onChange(of:identity) {_,_ in selection=nil;rowAxis=nil;columnAxis=nil;fixed=[:];fixedDrafts=[:];indexError=nil;heatmapMode=false}
    }
    private var metadata:some View {
        HStack(spacing:10) {
            if portChoices.count>1 {
                FlatChoiceControl(title:"Output port",selection:port,options:portChoices.map {.init(id:$0,title:$0)},compact:true,accent:accent,onSelect:onSelectPort)
                    .frame(width:105).accessibilityIdentifier("tensor.output-port")
            } else {Text(port).fontWeight(.semibold).foregroundStyle(accent)}
            Text(sample.shape.map {"["+$0.map(String.init).joined(separator:", ")+"]"} ?? "Shape unavailable")
            Text(sample.dtype)
            Text("\(sample.retainedCount) / \(sample.totalCount.map(String.init) ?? "?") retained")
                .foregroundStyle(MLTheme.muted)
            Button {showsEvidence.toggle()} label:{Image(systemName:"info.circle")}.buttonStyle(.plain)
                .accessibilityLabel("Captured tensor evidence")
                .popover(isPresented:$showsEvidence) {
                    VStack(alignment:.leading,spacing:8) {
                        Text(sample.availability).font(.caption.weight(.semibold))
                        if let evidenceLabel {Text(evidenceLabel).textSelection(.enabled)}
                        Text("Coordinates refer to the captured row-major values. This viewer retains at most 256 scalar rows; a matrix shows a bounded captured window. Unavailable slices are never queried.")
                        if !sample.axes.isEmpty {Text(sample.axes.joined(separator:" · "))}
                        Text("Shading compares only the displayed finite captured values. It does not describe the full tensor.")
                    }.font(.caption).foregroundStyle(MLTheme.muted).padding(14).frame(width:340)
                }
        }.font(.system(size:11,design:.monospaced)).lineLimit(1)
    }
    private var controls:some View {
        HStack(spacing:8) {
            Button("Values") {heatmapMode=false}.buttonStyle(FlatButtonStyle(prominent:!heatmapMode))
            Button("Heatmap") {heatmapMode=true}.buttonStyle(FlatButtonStyle(prominent:heatmapMode))
            if !heatmapMode {
            modeButton("Matrix",id:"matrix").disabled(!matrixEligible)
            modeButton("Coordinates",id:"coordinates")
            if matrixEligible,mode == "matrix",case .success(let value)=matrix,value.isPartialWindow {
                Text("\(value.rowIndices.count)×\(value.columnIndices.count) window").font(.caption2).foregroundStyle(MLTheme.muted)
            }
            Toggle("Cell shading",isOn:$shading).toggleStyle(.checkbox).font(.caption).tint(accent)
            Button("Copy cell") {
                guard let value=currentGrid?.selected(selection) else {return}
                NSPasteboard.general.clearContents();NSPasteboard.general.setString(value.text,forType:.string)
            }.buttonStyle(FlatButtonStyle()).disabled(currentGrid?.selected(selection) == nil)
            }
        }
    }
    private func modeButton(_ title:String,id:String)->some View {
        Button(title) {mode=id;selection=nil}.buttonStyle(.plain).font(.caption)
            .padding(.horizontal,9).padding(.vertical,5)
            .background((id == (matrixEligible ? mode:"coordinates")) ? accent.opacity(0.12):Color.clear)
            .overlay(Rectangle().stroke((id == (matrixEligible ? mode:"coordinates")) ? accent:MLTheme.rule))
    }
    private var sliceControls:some View {
        VStack(alignment:.leading,spacing:8) {
            HStack(spacing:10) {
                axisChoice("Row axis",selection:row,isRow:true)
                axisChoice("Column axis",selection:column,isRow:false)
                Spacer(minLength:0)
            }
            if !fixedIndices.isEmpty {
                ScrollView(.horizontal) {HStack(alignment:.top,spacing:12) {
                    ForEach(fixedIndices.keys.sorted(),id:\.self) {axis in
                        VStack(alignment:.leading,spacing:4) {
                            Text("Axis \(axis) index · 0…\(max(0,shape[axis]-1))").font(.caption2).foregroundStyle(MLTheme.muted)
                            HStack(spacing:4) {
                            Button {step(axis:axis,by:-1)} label:{Image(systemName:"chevron.left")}
                                .buttonStyle(FlatButtonStyle()).disabled((fixed[axis] ?? 0)<=0)
                                .accessibilityLabel("Previous matrix on axis \(axis)")
                            TextField("Fixed index",text:Binding(get:{fixedDrafts[axis] ?? String(fixed[axis] ?? 0)},set:{value in
                                fixedDrafts[axis]=value
                                guard let index=Int(value),shape[axis]>0,(0..<shape[axis]).contains(index) else {indexError="Axis \(axis) requires a captured index from 0 through \(max(0,shape[axis]-1)).";selection=nil;return}
                                fixed[axis]=index;indexError=nil;selection=nil
                            })).flatField().frame(width:65).accessibilityLabel("Fixed index for axis \(axis)")
                            Button {step(axis:axis,by:1)} label:{Image(systemName:"chevron.right")}
                                .buttonStyle(FlatButtonStyle()).disabled((fixed[axis] ?? 0)>=shape[axis]-1)
                                .accessibilityLabel("Next matrix on axis \(axis)")
                            }
                        }
                    }
                }}.frame(height:52)
            }
        }
    }
    private func step(axis:Int,by delta:Int) {
        let next=(fixed[axis] ?? 0)+delta
        guard shape.indices.contains(axis),(0..<shape[axis]).contains(next) else {return}
        fixed[axis]=next;fixedDrafts[axis]=String(next);indexError=nil;selection=nil
    }
    private func axisChoice(_ title:String,selection:Int,isRow:Bool)->some View {
        VStack(alignment:.leading,spacing:4) {
            Text(title).font(.caption2).foregroundStyle(MLTheme.muted)
            FlatChoiceControl(title:title,selection:String(selection),options:shape.indices.map {.init(id:String($0),title:"Axis \($0) · \(shape[$0])")},compact:true,accent:accent) {value in
                guard let next=Int(value),shape.indices.contains(next) else {return}
                if isRow {if next==column {columnAxis=row};rowAxis=next}
                else {if next==row {rowAxis=column};columnAxis=next}
                fixedDrafts=[:];indexError=nil;self.selection=nil
            }.frame(width:145)
        }
    }
    private var coordinateGrid:TensorCapturedGrid {
        .init(headers:["Coordinate","Value","State"],gutterTitle:"Offset",gutter:sample.rows.map {String($0.id)},cells:sample.rows.map {value in
            [.init(text:value.coordinates.map {$0.isEmpty ? "Scalar":"["+$0.map(String.init).joined(separator:", ")+"]"} ?? "Unknown",state:"Captured",numericValue:nil),
             .init(text:value.value,state:value.state,numericValue:value.numericValue),.init(text:value.state,state:value.state,numericValue:nil)]
        })
    }
    private func matrixGrid(_ value:TensorSampleInspection.Matrix)->TensorCapturedGrid {
        .init(headers:value.columnIndices.map {"\($0)"},gutterTitle:"Axis \(value.rowAxis)",gutter:value.rowIndices.map(String.init),cells:value.cells.map {$0.map {.init(text:$0.value,state:$0.state,numericValue:$0.numericValue)}})
    }
    private var currentGrid:TensorCapturedGrid? {
        if mode != "matrix" || !matrixEligible {return coordinateGrid}
        guard fixedIndexError == nil,case .success(let value)=matrix else {return nil}
        return matrixGrid(value)
    }
    private func grid(_ value:TensorCapturedGrid)->some View {
        TensorCapturedNativeGrid(grid:value,accent:accent,shading:shading,selection:$selection)
            .frame(minHeight:120,idealHeight:240,maxHeight:300)
    }
    private func unavailable(_ message:String)->some View {
        Text(message).font(.caption).foregroundStyle(MLTheme.muted).padding(12).frame(maxWidth:.infinity,minHeight:100,alignment:.topLeading)
    }
}
