import SwiftUI
import Charts

/// The same evidence-only chart is used by current Train Results and verified
/// History. Pausing freezes a view snapshot, never the worker or capture ledger.
struct TrainingMetricsChart:View {
    let metrics:[EpochMetric]
    var progress:[TrainingProgressPoint]=[]
    var observed:Int=0
    var recorded=false
    var running=false
    var large=false
    var trainingControl:TrainingControlReport?=nil
    var selfPlay=false
    private var unit:String {selfPlay ? "Iteration":"Epoch"}
    private func axisTitle(_ a:TrainingChartAxis)->String {a == .epoch ? unit:a.rawValue}
    @State private var axis:TrainingChartAxis = .updates
    @State private var metric:TrainingChartMetric = .loss
    @State private var smoothed=false
    @State private var training=true
    @State private var validation=true
    @State private var viewport=TrainingChartViewport()
    @State private var projections=BoundedPresentationCache<TrainingChartInputs,TrainingChartProjection>(maximumEntries:1,maximumCost:16_000_000)
    @Environment(\.dismiss) private var dismiss
    @Environment(\.trainingChartViewportHeight) private var viewportHeight
    @State private var selectedX:Double?
    @State private var frozen:Snapshot?
    @State private var expanded=false
    @State private var information=false
    @State private var measurements=false
    @State private var selectedRow:Int?
    @State private var reportPolicy=TrainingReportingPolicy.current
    private struct Snapshot {let metrics:[EpochMetric];let progress:[TrainingProgressPoint];let observed:Int}
    private var display:Snapshot {frozen ?? .init(metrics:metrics,progress:progress,observed:observed)}
    private var hasAccuracy:Bool {display.metrics.contains(where:{$0.accuracy != nil}) || display.progress.contains(where:{$0.accuracy != nil})}
    private var effectiveAxis:TrainingChartAxis {display.progress.isEmpty ? .epoch:axis}
    private var projection:TrainingChartProjection {
        let snapshot=display
        let input=TrainingChartInputs(metrics:snapshot.metrics,progress:snapshot.progress,axis:effectiveAxis,metric:metric,smoothed:smoothed,training:training,validation:validation,viewport:viewport)
        return projections.value(key:"chart",input:input,cost:max(1,(snapshot.metrics.count+snapshot.progress.count)*512)) {TrainingChartProjection(input)}
    }
    private var raw:[TrainingChartValue] {projection.raw}
    private var domain:ClosedRange<Double> {projection.domain}
    private var range:ClosedRange<Double> {projection.range}
    private var visible:[TrainingChartValue] {projection.visible}
    private var selected:TrainingChartValue? {guard let selectedX else {return nil};return visible.min {abs($0.x-selectedX)<abs($1.x-selectedX)}}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            header
            controls
            if raw.isEmpty {
                Text(running ? "Waiting for a completed training batch…":"No captured measurements").font(.caption).foregroundStyle(MLTheme.muted).frame(maxWidth:.infinity,minHeight:180)
            } else {
                plot(height:TrainingChartLayout.plotHeight(availableHeight:viewportHeight,expanded:large))
                navigator
                readout
                if let trainingControl {Text("\(trainingControl.stopReason=="earlyStopping" ? "Stopped early":"Completed") · \(trainingControl.completedEpoch) epochs · Selected epoch \(trainingControl.selectedEpoch) · Validation loss \(String(format:"%.5g",trainingControl.validationLoss))").font(.caption2).foregroundStyle(MLTheme.muted)}
                if recorded {DisclosureGroup("Measurements",isExpanded:$measurements) {if measurements {measurementTable.frame(minHeight:130,idealHeight:190,maxHeight:240)}}.font(.caption)}
            }
        }.padding(12).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule)).windowActivatingControls()
            .onChange(of:running) {_,value in if !value {frozen=nil}}
            .onChange(of:hasAccuracy) {_,value in if !value {metric = .loss}}
            .onChange(of:effectiveAxis) {_,_ in viewport.fit();selectedX=nil}
            .onChange(of:metric) {_,_ in selectedX=nil}
            .sheet(isPresented:$expanded) {
                GeometryReader {size in
                    ThemedScrollView {TrainingMetricsChart(metrics:display.metrics,progress:display.progress,observed:display.observed,recorded:recorded,running:running,large:true,trainingControl:trainingControl,selfPlay:selfPlay)
                        .environment(\.trainingChartViewportHeight,size.size.height-40).padding(20)}
                }.frame(minWidth:700,idealWidth:900,minHeight:500,idealHeight:650)
            }
    }
    private var header:some View {
        HStack(spacing:8) {
            Circle().fill(running ? MLTheme.rose:MLTheme.muted).frame(width:6,height:6)
            Text(recorded ? "Recorded training":"Training").font(.system(size:12,weight:.semibold))
            if let last=display.progress.last {Text("\(unit) \(last.epoch) · \(selfPlay ? "update":"batch") \(last.epochBatch)/\(last.epochBatches)").font(.caption).foregroundStyle(MLTheme.muted)}
            else if let last=display.metrics.last {Text("Epoch \(last.epoch)").font(.caption).foregroundStyle(MLTheme.muted)}
            Spacer(minLength:4)
            if running {Button(frozen==nil ? "Pause chart":"Resume chart") {frozen=frozen==nil ? display:nil}.buttonStyle(FlatButtonStyle()).accessibilityIdentifier("training.chart.pause")}
            if !large {Button {expanded=true} label:{Image(systemName:"arrow.up.left.and.arrow.down.right")}.buttonStyle(FlatButtonStyle()).help("Expand training chart")}
            Button {information.toggle()} label:{Image(systemName:"info.circle")}.buttonStyle(FlatButtonStyle()).help("Chart information and reporting")
                .popover(isPresented:$information) {informationPanel.padding(16).frame(width:320)}
            if large {Button {dismiss()} label:{Label("Close",systemImage:"xmark")}
                .buttonStyle(FlatButtonStyle()).keyboardShortcut(.cancelAction)
                .accessibilityLabel("Close chart").accessibilityIdentifier("training.chart.close")}
        }
    }
    private var controls:some View {
        ViewThatFits(in:.horizontal) {
            HStack(spacing:14) {selectors;series;displayControls}
            VStack(alignment:.leading,spacing:8) {selectors;HStack {series;Spacer();displayControls}}
        }.font(.caption).controlSize(.small)
    }
    private var selectors:some View {
        HStack(spacing:10) {
            FlatChoiceControl(title:"Metric",selection:metric.rawValue,options:([TrainingChartMetric.loss]+(hasAccuracy ? [.accuracy]:[])+(display.progress.contains(where:{$0.moe != nil}) ? [.routing]:[])).map {.init(id:$0.rawValue,title:$0.rawValue)},compact:true,accent:MLTheme.rose) {if let value=TrainingChartMetric(rawValue:$0) {metric=value;if value != .loss{smoothed=false}}}.frame(width:120)
            FlatChoiceControl(title:"Axis",selection:effectiveAxis.rawValue,options:(display.progress.isEmpty ? [.epoch]:selfPlay ? [.updates,.epoch,.time]:TrainingChartAxis.allCases).map {.init(id:$0.rawValue,title:axisTitle($0))},compact:true,accent:MLTheme.rose) {if let value=TrainingChartAxis(rawValue:$0) {axis=value}}.frame(width:120)
        }.labelsHidden()
    }
    private var series:some View {
        HStack(spacing:10) {
            Toggle("Training",isOn:$training).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.muted)).disabled(metric != .loss)
            if !selfPlay && metric != .routing {Toggle("Validation",isOn:$validation).toggleStyle(FileInputCheckboxStyle(tint:MLTheme.rose))}
        }
    }
    private var displayControls:some View {
        HStack(spacing:5) {
            Button("Raw") {smoothed=false}.buttonStyle(FlatButtonStyle(prominent:!smoothed))
            Button("Smoothed") {smoothed=true}.buttonStyle(FlatButtonStyle(prominent:smoothed)).disabled(metric != .loss)
            Button("Fit") {viewport.fit();selectedX=nil}.buttonStyle(FlatButtonStyle())
                .disabled(viewport.isFitted && selectedX==nil).help("Show all captured measurements")
                .accessibilityIdentifier("training.chart.fit")
        }
    }
    private func plot(height:CGFloat)->some View {
        let rows=visible
        let lastTrainingID=rows.last(where:{$0.series=="Training"})?.id
        return Chart {
            ForEach(projection.plot) {row in
                if metric == .routing {
                    LineMark(x:.value(effectiveAxis.rawValue,row.x),y:.value(metric.rawValue,row.value),series:.value("Series",row.series)).foregroundStyle(routingColor(row.series))
                } else if row.series=="Training" {
                    LineMark(x:.value(effectiveAxis.rawValue,row.x),y:.value(metric.rawValue,row.value),series:.value("Series","Training")).foregroundStyle(MLTheme.muted)
                    if row.id==lastTrainingID {PointMark(x:.value(effectiveAxis.rawValue,row.x),y:.value(metric.rawValue,row.value)).foregroundStyle(MLTheme.muted).symbolSize(18)}
                }
                else {
                    LineMark(x:.value(effectiveAxis.rawValue,row.x),y:.value(metric.rawValue,row.value),series:.value("Series","Validation")).foregroundStyle(MLTheme.rose)
                    PointMark(x:.value(effectiveAxis.rawValue,row.x),y:.value(metric.rawValue,row.value)).foregroundStyle(MLTheme.rose).symbolSize(15)
                }
            }
            if let selected {RuleMark(x:.value(effectiveAxis.rawValue,selected.x)).foregroundStyle(MLTheme.muted.opacity(0.5)).lineStyle(StrokeStyle(lineWidth:1,dash:[3,3]))}
        }.chartXScale(domain:range).chartYScale(domain:metric == .accuracy ? 0...1:lossDomain)
            .chartXAxisLabel(effectiveAxis == .time ? "Elapsed seconds":effectiveAxis.rawValue)
            .chartYAxisLabel(metric == .routing ? "Task loss + weighted routing penalties":metric == .loss ? (selfPlay ? "Captured update loss":display.progress.isEmpty ? "Epoch loss":"Interval loss · validation at epoch end"):"Validation accuracy")
            .chartXSelection(value:$selectedX)
            .chartGesture { proxy in
                DragGesture(minimumDistance:0)
                    .onChanged {proxy.selectXValue(at:$0.location.x)}
                    .onEnded {proxy.selectXValue(at:$0.location.x)}
            }.frame(height:height)
            .accessibilityLabel(recorded ? "Recorded training chart":"Live training chart")
    }
    private func routingColor(_ series:String)->Color {switch series{case "Task loss":return MLTheme.muted;case "Balance penalty":return .blue;case "Router z penalty":return MLTheme.teal;default:return MLTheme.rose}}
    private var lossDomain:ClosedRange<Double> {projection.lossDomain}
    private var navigator:some View {
        VStack(spacing:2) {
            Chart(projection.navigator) {row in
                LineMark(x:.value("Position",row.x),y:.value("Value",row.value),series:.value("Series",row.series)).foregroundStyle(metric == .routing ? routingColor(row.series).opacity(0.7):(row.series=="Training" ? MLTheme.muted.opacity(0.5):MLTheme.rose.opacity(0.7)))
            }.chartXScale(domain:domain).chartXAxis(.hidden).chartYAxis(.hidden).frame(height:28).overlay {
                GeometryReader {geo in Rectangle().fill(MLTheme.rose.opacity(0.12)).overlay(Rectangle().stroke(MLTheme.rose.opacity(0.6))).frame(width:max(1,geo.size.width*(viewport.upper-viewport.lower))).offset(x:geo.size.width*viewport.lower)}
            }
            HStack(spacing:8) {
                Slider(value:Binding(get:{viewport.lower},set:{viewport.setLower($0)}),in:0...1).tint(MLTheme.rose).accessibilityLabel("Visible range start")
                Slider(value:Binding(get:{viewport.upper},set:{viewport.setUpper($0)}),in:0...1).tint(MLTheme.rose).accessibilityLabel("Visible range end")
                Text("\(format(range.lowerBound))–\(format(range.upperBound))").monospacedDigit().font(.caption2).foregroundStyle(MLTheme.muted)
            }
        }
    }
    private var readout:some View {
        VStack(alignment:.leading,spacing:4) {
            if let selected {
                Text("\(selected.series) · \(format(selected.value)) · \(axisTitle(effectiveAxis)) \(format(selected.x)) · \(unit.lowercased()) \(selected.epoch)").font(.caption).monospacedDigit()
            } else {
                HStack(spacing:16) {
                    if let last=raw.last(where:{$0.series=="Training"}) {Text("Training \(format(last.value))")}
                    if let last=raw.last(where:{$0.series=="Validation"}) {Text("Validation \(format(last.value))")}
                    if let best=display.metrics.min(by:{$0.validationLoss<$1.validationLoss}),metric == .loss {Text("Best validation \(format(best.validationLoss)) · epoch \(best.epoch)")}
                }.font(.caption).monospacedDigit().foregroundStyle(MLTheme.muted)
            }
            if metric == .routing {Text("Task loss · Balance penalty · Router z penalty · Training objective").font(.caption2).foregroundStyle(MLTheme.muted)}
            if display.observed>display.progress.count {Text("\(display.progress.count) retained observations from \(display.observed); chart sampled.").font(.caption2).foregroundStyle(MLTheme.muted)}
            if frozen != nil {Text("Chart paused · training and capture continue").font(.caption2).foregroundStyle(MLTheme.rose)}
        }
    }
    private var measurementTable:some View {
        CapturedNativeTable(columns:WorkflowResultTableData.columns([axisTitle(effectiveAxis),unit,"Series",metric.rawValue],numeric:[0,1,3]),rows:raw.enumerated().map {i,row in WorkflowResultTableData.row(i,[format(row.x),String(row.epoch),row.series,format(row.value)])},selectedRow:$selectedRow,rowHeaderTitle:"#",fillsWidth:true)
    }
    private var informationPanel:some View {
        VStack(alignment:.leading,spacing:12) {
            Text("Training measurements").font(.headline)
            if metric == .routing {Text("Task loss, weighted balance penalty, weighted router z penalty and their sum are captured over completed intervals. Routing regularization applies only to Training; Validation selection uses task metrics.")}
            if selfPlay {Text("Self-play plots captured completed updates. Iteration marks the collection/update boundary; policy/value components are selected above the chart. Evaluation is independent and appears in standard tables. Pause chart affects display only. Exact CPU resume uses committed iteration state.")} else {
            Text("Training loss covers completed batches since the previous report, weighted by the objective’s actual denominator. Validation and accuracy appear only after evaluation. Time is elapsed wall time at completed work boundaries. Smoothed uses an exponential average (20% new value) for Training only. Raw values, checkpoints and best validation remain unchanged.")
            Text("Pause freezes this display. History detail capture controls retention. Long runs keep at most 8,192 sampled progress observations; epoch metrics remain exact. Resume remains at an epoch boundary.")
            }
            if !recorded && !selfPlay {
                Divider()
                Text("Reporting · next run").font(.subheadline.weight(.semibold))
                Text("Every completed batches").foregroundStyle(MLTheme.muted)
                FlatChoiceControl(title:"Reporting batches",selection:String(reportPolicy.everyBatches),options:[1,5,10,25,50,100,1000].map {.init(id:String($0),title:String($0))},compact:true,accent:MLTheme.rose) {if let value=Int($0) {reportPolicy.everyBatches=value}}
                Text("Or after elapsed seconds").foregroundStyle(MLTheme.muted)
                FlatChoiceControl(title:"Reporting seconds",selection:String(reportPolicy.everySeconds),options:[0.2,0.5,1.0,2.0,5.0,10.0,60.0].map {.init(id:String($0),title:format($0))},compact:true,accent:MLTheme.rose) {if let value=Double($0) {reportPolicy.everySeconds=value}}
                Text("Whichever comes first, after a batch completes. The first and final batches always report. These preferences affect observations, not training.").foregroundStyle(MLTheme.muted)
            }
        }.font(.caption).tint(MLTheme.rose).onChange(of:reportPolicy) {_,value in value.save()}
    }
    private func format(_ value:Double)->String {String(format:"%.5g",value)}
}
