import Foundation

struct TrainingChartInputs:Equatable {
    let metrics:[EpochMetric],progress:[TrainingProgressPoint]
    let axis:TrainingChartAxis,metric:TrainingChartMetric
    let smoothed:Bool,training:Bool,validation:Bool
    let viewport:TrainingChartViewport
}
/// Cached derived arrays exclude pointer selection and available height, so
/// inspecting/resizing a chart does not repeatedly project the full ledger.
struct TrainingChartProjection {
    let raw:[TrainingChartValue],visible:[TrainingChartValue],plot:[TrainingChartValue],navigator:[TrainingChartValue]
    let domain:ClosedRange<Double>,range:ClosedRange<Double>,lossDomain:ClosedRange<Double>
    init(_ input:TrainingChartInputs) {
        raw=TrainingChartModel.values(metrics:input.metrics,progress:input.progress,axis:input.axis,metric:input.metric,smoothed:false)
        domain=TrainingChartModel.domain(raw)
        range=TrainingChartModel.window(domain,lower:input.viewport.lower,upper:input.viewport.upper)
        let values=input.smoothed ? TrainingChartModel.values(metrics:input.metrics,progress:input.progress,axis:input.axis,metric:input.metric,smoothed:true):raw
        let window=range
        visible=values.filter {window.contains($0.x) && (input.metric == .routing ? input.training:(($0.series=="Training" && input.training) || ($0.series=="Validation" && input.validation)))}
        plot=TrainingChartModel.render(visible)
        navigator=TrainingChartModel.render(raw,limit:150)
        let low=visible.map(\.value).min() ?? 0,high=visible.map(\.value).max() ?? 1
        let pad=max(abs(high-low)*0.08,max(abs(high)*0.01,0.000001))
        lossDomain=(low-pad)...(high+pad)
    }
}
