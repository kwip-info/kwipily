/// Selects observed values only. No smoothing or synthetic interpolation of metrics.
enum MetricSampling {
    static func sample<Element>(_ values:[Element],limit:Int=250)->[Element] {
        guard limit > 0 else { return [] }
        guard values.count > limit else { return values }
        guard limit > 1 else { return [values[values.count-1]] }
        let span=values.count-1, intervals=limit-1
        // Quotient/remainder form avoids multiplying the full collection count.
        return (0..<limit).map { index in
            values[(span / intervals) * index + ((span % intervals) * index) / intervals]
        }
    }
    static func shouldPublish(epoch:Int,totalEpochs:Int)->Bool {
        guard totalEpochs > 0, epoch >= 1, epoch <= totalEpochs else { return false }
        let stride=(totalEpochs-1)/200+1
        return epoch == 1 || epoch == totalEpochs || epoch % stride == 0
    }
}
