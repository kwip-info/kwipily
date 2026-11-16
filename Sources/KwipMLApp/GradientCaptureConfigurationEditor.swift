import SwiftUI
import KwipMLCore

/// Shared configuration UI; callers own persistence, runtime support and draft gates.
struct GradientCaptureConfigurationEditor:View {
    let configuration:GradientCaptureConfiguration?
    let canCapture:Bool
    let isBusy:Bool
    let origin:String
    let currentOrigin:()->String?
    var onDraftChange:((String,Bool)->Void)?=nil
    let apply:(GradientCaptureConfiguration?)->Bool
    @State private var rejection=false
    private var config:GradientCaptureConfiguration {configuration ?? .init()}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            SectionLabel(title:"Gradient measurements")
            Button {
                guard currentOrigin() == origin else {rejection=true;return}
                rejection = !apply(configuration == nil ? .init() : nil)
            } label: {
                Label("Capture gradient summaries",systemImage:configuration == nil ? "square":"checkmark.square.fill")
            }.buttonStyle(FlatButtonStyle()).disabled(isBusy || (configuration == nil && !canCapture))
            Text("Observe gradients after backward and before the optimizer update. Adds computation and device synchronization; does not clip gradients or change the optimizer.").font(.caption).foregroundStyle(MLTheme.muted)
            if !canCapture {Text("Gradient capture is unavailable until this runtime reports compatible support.").font(.caption).foregroundStyle(MLTheme.muted)}
            if configuration != nil {
                parameter("Capture every N epochs",key:"everyEpochs",value:config.everyEpochs,bounds:GradientCaptureConfiguration.epochCadenceBounds) {$0.everyEpochs=$1}
                parameter("Batches per captured epoch",key:"maxBatchesPerEpoch",value:config.maxBatchesPerEpoch,bounds:GradientCaptureConfiguration.batchBounds) {$0.maxBatchesPerEpoch=$1}
                parameter("Total batch capture budget",key:"maxCaptures",value:config.maxCaptures,bounds:GradientCaptureConfiguration.captureBounds) {$0.maxCaptures=$1}
                Text("Stores counts, L2 norm and maximum absolute gradient only. No raw gradient arrays or source rows are retained. Resume starts a new capture budget; parent gradient history is not imported.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            if rejection {Text("This gradient configuration was not applied.").font(.caption).foregroundStyle(MLTheme.rose)}
        }
    }
    private func parameter(_ title:String,key:String,value:Int,bounds:ClosedRange<Int>,update:@escaping(inout GradientCaptureConfiguration,Int)->Void)->some View {
        VStack(alignment:.leading,spacing:4) {
            Text(title).font(.caption)
            BoundedNumberField(title:title,value:.integer(Int64(value)),domain:.integer(minimum:Double(bounds.lowerBound),maximum:Double(bounds.upperBound)),origin:origin+":"+key,currentOrigin:{currentOrigin().map {$0+":"+key}},onDraftChange:onDraftChange) {numeric in
                guard case .integer(let integer)=numeric else {return false}
                var next=config;update(&next,Int(integer))
                return (try? next.validate()) != nil && apply(next)
            }
        }.disabled(isBusy)
    }
}
