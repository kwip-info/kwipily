import SwiftUI
import KwipMLCore

private struct ConfigureTensorCaptureKey:EnvironmentKey {static let defaultValue:(()->Void)?=nil}
extension EnvironmentValues {
    var configureTensorCapture:(()->Void)? {get {self[ConfigureTensorCaptureKey.self]} set {self[ConfigureTensorCaptureKey.self]=newValue}}
}

/// All colors and readouts derive from authenticated tile aggregates. No source
/// query, model inference, interpolation or unbounded image allocation occurs here.
struct TensorHeatmapView:View {
    let summary:TensorHeatmapSummary
    let accent:Color
    var large=false
    var timelineRange:ClosedRange<Double>?=nil
    @Environment(\.dismiss) private var dismiss
    @State private var planeIndex=0
    @State private var overview=false
    @State private var transpose=false
    @State private var aggregation="Mean"
    @State private var scale="Symmetric"
    @State private var limit="1"
    @State private var selected:Int?
    @State private var expanded=false
    @State private var information=false
    init(summary:TensorHeatmapSummary,accent:Color,large:Bool=false,timelineRange:ClosedRange<Double>?=nil,initialPlane:Int=0,initialOverview:Bool=false,initialTranspose:Bool=false,initialAggregation:String="Mean",initialScale:String="Symmetric",initialLimit:String="1",initialSelected:Int?=nil) {
        self.summary=summary;self.accent=accent;self.large=large;self.timelineRange=timelineRange
        _planeIndex=State(initialValue:min(initialPlane,max(0,summary.planes.count-1)))
        _overview=State(initialValue:initialOverview);_transpose=State(initialValue:initialTranspose)
        _aggregation=State(initialValue:initialAggregation);_scale=State(initialValue:initialScale)
        _limit=State(initialValue:initialLimit);_selected=State(initialValue:initialSelected)
    }
    private var plane:TensorHeatmapSummary.Plane? {summary.planes.indices.contains(planeIndex) ? summary.planes[planeIndex]:nil}
    private var grid:[Int] {summary.gridShape ?? [0,0]}
    private var shape:[Int] {summary.planeShape ?? [0,0]}
    private var plotRows:Int {transpose ? grid[1]:grid[0]}
    private var plotColumns:Int {transpose ? grid[0]:grid[1]}
    private var fixedLimit:Double? {guard let n=Double(limit),n.isFinite,n>0 else {return nil};return n}
    private var scaleRange:ClosedRange<Double> {
        let values=(overview ? summary.planes:plane.map {[$0]} ?? []).flatMap {measurements($0).compactMap {$0}}
        let low=values.min() ?? 0,high=values.max() ?? 0
        if scale=="Fixed",let n=fixedLimit {return -n...n}
        if scale=="Symmetric" {let n=max(timelineRange.map {max(abs($0.lowerBound),abs($0.upperBound))} ?? 0,abs(low),abs(high),1e-12);return -n...n}
        if let timelineRange,scale=="Automatic" {return timelineRange}
        if low<high {return low...high}
        let extent=max(abs(low),abs(high),1e-12)
        return -extent...extent
    }
    private var axisDescription:String {
        guard !summary.sourceShape.isEmpty else {return "Scalar"}
        guard summary.sourceShape.count>=2 else {return "One display row · Columns: Axis 0"}
        return "Rows: Axis \(transpose ? summary.columnAxis ?? 0:summary.rowAxis ?? 0) · Columns: Axis \(transpose ? summary.rowAxis ?? 0:summary.columnAxis ?? 0)"
    }
    private func measurements(_ p:TensorHeatmapSummary.Plane)->[Double?] {aggregation=="Minimum" ? p.minimum:aggregation=="Maximum" ? p.maximum:p.mean}
    private func number(_ n:Double?)->String {n.map {String(format:"%.5g",$0)} ?? "Unavailable"}
    var body:some View {
        let range=scaleRange
        VStack(alignment:.leading,spacing:8) {
            ViewThatFits(in:.horizontal) {
                HStack {modeControls;Spacer();planeControls}
                VStack(alignment:.leading,spacing:8) {modeControls;planeControls}
            }
            ViewThatFits(in:.horizontal) {
                HStack(spacing:12) {measureControls;scaleControls}
                VStack(alignment:.leading,spacing:8) {measureControls;scaleControls}
            }
            if scale=="Fixed",fixedLimit==nil {Text("Enter a finite positive color limit.").font(.caption).foregroundStyle(.orange)}
            if overview {
                LazyVGrid(columns:[GridItem(.adaptive(minimum:100,maximum:180))],spacing:12) {
                    ForEach(summary.planes.indices,id:\.self) {index in
                        Button {planeIndex=index;overview=false;selected=nil} label:{
                            VStack(alignment:.leading,spacing:4) {
                                Text(planeLabel(summary.planes[index])).font(.caption2).lineLimit(1)
                                heatmap(summary.planes[index],interactive:false,range:range).frame(height:90)
                            }.padding(5).overlay(Rectangle().stroke(index==planeIndex ? accent:MLTheme.rule))
                        }.buttonStyle(.plain).accessibilityLabel("Inspect "+planeLabel(summary.planes[index]))
                    }
                }.padding(2)
            } else if let plane {
                HStack(spacing:8) {
                    VStack(alignment:.leading,spacing:3) {
                        Text(axisDescription).font(.caption2).foregroundStyle(MLTheme.muted)
                        heatmap(plane,interactive:true,range:range).frame(height:large ? 360:220)
                        HStack {Text("0");Spacer();Text(String(max(0,(transpose ? shape[0]:shape[1])-1)))}.font(.caption2.monospacedDigit()).foregroundStyle(MLTheme.muted)
                    }
                    legend(range:range).frame(width:65)
                }
                if let selected,plane.mean.indices.contains(selected) {selectionReadout(plane,index:selected)}
            }
            Text(coverage).font(.caption2).foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true)
        }.onChange(of:summary.sourceShape) {_,_ in planeIndex=min(planeIndex,max(0,summary.planes.count-1));selected=nil}
        .onChange(of:summary.gridShape) {_,_ in selected=nil}
        .onChange(of:summary.planes.count) {_,count in planeIndex=min(planeIndex,max(0,count-1));selected=nil}
        .padding(10).sheet(isPresented:$expanded) {
            ThemedScrollView {
                TensorHeatmapView(summary:summary,accent:accent,large:true,timelineRange:timelineRange,initialPlane:planeIndex,initialOverview:overview,initialTranspose:transpose,initialAggregation:aggregation,initialScale:scale,initialLimit:limit,initialSelected:selected).padding(12)
            }
                .frame(minWidth:620,idealWidth:850,minHeight:430,idealHeight:620)
        }
    }
    private var modeControls:some View {
        HStack(spacing:5) {
            Button("Plane") {overview=false}.buttonStyle(FlatButtonStyle(prominent:!overview))
            Button("Overview") {overview=true}.buttonStyle(FlatButtonStyle(prominent:overview))
            Button {information=true} label:{Image(systemName:"info.circle")}.buttonStyle(FlatButtonStyle()).accessibilityLabel("Heatmap evidence and coverage")
                .popover(isPresented:$information) {
                    VStack(alignment:.leading,spacing:8) {
                        Text("Captured aggregate values").fontWeight(.semibold)
                        Text("Each tile summarizes its actual source coordinates. Mean uses finite values only; the readout reports non-finite counts separately. Empty or overflowed means remain unavailable. Colors may clip outside the chosen range; measurements do not change.")
                        Text("The final two captured axes define planes. Overview uses one shared scale. Partial plane coverage is explicit; unretained planes are never queried. Integer summaries use approximate floating-point values; Values retains exact captured integer text.")
                    }.font(.caption).padding(14).frame(width:340)
                }
            if large {Button("Close") {dismiss()}.buttonStyle(FlatButtonStyle()).keyboardShortcut(.cancelAction)}
            else {Button {expanded=true} label:{Image(systemName:"arrow.up.left.and.arrow.down.right")}.buttonStyle(FlatButtonStyle()).accessibilityLabel("Expand heatmap")}
        }.font(.caption)
    }
    private var planeControls:some View {
        HStack(spacing:5) {
            Button {planeIndex=max(0,planeIndex-1);selected=nil} label:{Image(systemName:"chevron.left")}.disabled(planeIndex==0).accessibilityLabel("Previous captured plane")
            FlatChoiceControl(title:"Plane",selection:String(planeIndex),options:summary.planes.map {.init(id:String($0.index),title:planeLabel($0))},compact:true,accent:accent) {if let n=Int($0) {planeIndex=n;selected=nil}}
                .frame(minWidth:120,maxWidth:230)
            Button {planeIndex=min(summary.planes.count-1,planeIndex+1);selected=nil} label:{Image(systemName:"chevron.right")}.disabled(planeIndex+1>=summary.planes.count).accessibilityLabel("Next captured plane")
            Button {transpose.toggle();selected=nil} label:{Image(systemName:"arrow.triangle.swap")}.accessibilityLabel("Swap heatmap row and column axes").disabled(summary.sourceShape.count<2)
        }.buttonStyle(FlatButtonStyle()).font(.caption)
    }
    private var measureControls:some View {
        FlatChoiceControl(title:"Tile value",selection:aggregation,options:["Mean","Minimum","Maximum"].map {.init(id:$0,title:$0)},compact:true,accent:accent) {aggregation=$0}
            .frame(width:175)
    }
    private var scaleControls:some View {
        HStack(spacing:8) {
            FlatChoiceControl(title:"Color scale",selection:scale,options:["Automatic","Symmetric","Fixed"].map {.init(id:$0,title:$0)},compact:true,accent:accent) {scale=$0}.frame(width:190)
            if scale=="Fixed" {Text("±");TextField("Limit",text:$limit).flatField().frame(width:75).accessibilityLabel("Fixed heatmap color limit")}
        }.font(.caption)
    }
    private func legend(range:ClosedRange<Double>)->some View {
        VStack(alignment:.leading,spacing:3) {
            Text(number(range.upperBound))
            LinearGradient(colors:[Color(red:0.83,green:0.48,blue:0.59),Color(red:0.12,green:0.12,blue:0.15),Color(red:0.20,green:0.42,blue:0.77)],startPoint:.top,endPoint:.bottom).frame(width:16).frame(maxHeight:.infinity)
            Text(number(range.lowerBound))
        }.font(.caption2.monospacedDigit())
    }
    private func color(_ value:Double?,range:ClosedRange<Double>)->Color {
        guard let value else {return Color.gray.opacity(0.3)}
        let magnitude=max(abs(range.lowerBound),abs(range.upperBound),1e-12)
        let t=scale=="Automatic" ? 2*min(1,max(0,(value/magnitude-range.lowerBound/magnitude)/(range.upperBound/magnitude-range.lowerBound/magnitude)))-1:min(1,max(-1,value/magnitude))
        let amount=abs(t)
        return Color(red:0.12+(t<0 ? 0.08:0.71)*amount,green:0.12+(t<0 ? 0.30:0.36)*amount,blue:0.15+(t<0 ? 0.62:0.44)*amount)
    }
    private func heatmap(_ plane:TensorHeatmapSummary.Plane,interactive:Bool,range:ClosedRange<Double>)->some View {
        let values=measurements(plane)
        return GeometryReader {size in
            Canvas {context,dimensions in
                let w=dimensions.width/CGFloat(plotColumns),h=dimensions.height/CGFloat(plotRows)
                for index in values.indices {
                    let sourceRow=index/grid[1],sourceColumn=index%grid[1]
                    let row=transpose ? sourceColumn:sourceRow,col=transpose ? sourceRow:sourceColumn
                    let rect=CGRect(x:CGFloat(col)*w,y:CGFloat(row)*h,width:w+0.25,height:h+0.25)
                    context.fill(Path(rect),with:.color(color(values[index],range:range)))
                    if values[index]==nil {context.stroke(Path(rect),with:.color(.gray.opacity(0.4)),lineWidth:0.5)}
                    if interactive,selected==index {context.stroke(Path(rect.insetBy(dx:0.7,dy:0.7)),with:.color(.white),lineWidth:1.5)}
                }
            }.contentShape(Rectangle()).gesture(SpatialTapGesture().onEnded {point in
                guard interactive,size.size.width>0,size.size.height>0 else {return}
                let row=min(plotRows-1,max(0,Int(point.location.y/size.size.height*CGFloat(plotRows))))
                let col=min(plotColumns-1,max(0,Int(point.location.x/size.size.width*CGFloat(plotColumns))))
                selected=transpose ? col*grid[1]+row:row*grid[1]+col
            })
        }.focusable(interactive).onMoveCommand {direction in
            guard interactive,plotRows>0,plotColumns>0 else {return}
            let old=selected ?? 0
            var row=transpose ? old%grid[1]:old/grid[1]
            var column=transpose ? old/grid[1]:old%grid[1]
            switch direction {case .left:column=max(0,column-1);case .right:column=min(plotColumns-1,column+1);case .up:row=max(0,row-1);case .down:row=min(plotRows-1,row+1);default:break}
            selected=transpose ? column*grid[1]+row:row*grid[1]+column
        }.overlay(Rectangle().stroke(MLTheme.rule)).accessibilityLabel(planeLabel(plane)+" heatmap").accessibilityValue(coverage)
    }
    private func planeLabel(_ plane:TensorHeatmapSummary.Plane)->String {
        plane.fixedIndices.isEmpty ? "Plane 1":plane.fixedIndices.enumerated().map {"Axis \($0.offset): \($0.element)"}.joined(separator:" · ")
    }
    private var coverage:String {
        "\(summary.planes.count) / \(summary.totalPlanes ?? 0) planes · \(shape[0]) × \(shape[1]) source values per plane → \(grid[0]) × \(grid[1]) aggregate tiles"+(summary.coverage=="partialPlanes" ? " · Partial plane capture":"")
    }
    private func selectionReadout(_ p:TensorHeatmapSummary.Plane,index:Int)->some View {
        let r=index/grid[1],c=index%grid[1]
        let y0=(r*shape[0]+grid[0]-1)/grid[0],y1=((r+1)*shape[0]+grid[0]-1)/grid[0]-1
        let x0=(c*shape[1]+grid[1]-1)/grid[1],x1=((c+1)*shape[1]+grid[1]-1)/grid[1]-1
        let coordinates=summary.sourceShape.isEmpty ? "Scalar":summary.sourceShape.count==1 ? "Axis 0: \(x0)–\(x1)":"Axis \(summary.rowAxis ?? 0): \(y0)–\(y1) · Axis \(summary.columnAxis ?? 0): \(x0)–\(x1)"
        return Text("\(coordinates)   Mean \(number(p.mean[index]))   Min \(number(p.minimum[index]))   Max \(number(p.maximum[index]))   Finite \(p.finiteCount[index])   NaN \(p.nanCount[index])   +∞ \(p.positiveInfinityCount[index])   −∞ \(p.negativeInfinityCount[index])")
            .font(.caption2.monospacedDigit()).textSelection(.enabled).fixedSize(horizontal:false,vertical:true)
    }
}
