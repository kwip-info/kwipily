import SwiftUI
import Charts

/// Axis bounds and labels derive from captured epochs, including resumed/sparse runs.
/// The chart never fabricates an epoch-zero measurement.
struct EpochChartAxis {
    let domain:ClosedRange<Double>
    let ticks:[Int]
    init?(epochs:[Int]) {
        let observed=Array(Set(epochs)).sorted()
        guard let first=observed.first,let last=observed.last else {return nil}
        domain=first == last ? (Double(first)-0.5)...(Double(last)+0.5) : Double(first)...Double(last)
        ticks=MetricSampling.sample(observed,limit:6)
    }
}
extension View {
    func capturedEpochAxis(_ axis:EpochChartAxis?)->some View {
        self.chartXScale(domain:axis?.domain ?? 0...1)
            .chartXAxis {
                AxisMarks(values:axis?.ticks ?? []) {value in
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel(anchor:axis?.ticks.count == 1 ? .top : value.index == 0 ? .topLeading : value.index == value.count-1 ? .topTrailing:.top) {
                        if let epoch=value.as(Int.self) {Text(String(epoch))}
                    }
                }
            }
    }
}
