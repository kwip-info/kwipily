import SwiftUI
import KwipMLCore

struct TensorHeatmapPlaybackView:View {
    let final:TensorHeatmapSummary
    let playback:TensorHeatmapPlayback?
    let accent:Color
    @State private var index:Int?=nil
    @State private var playing=false
    private var frames:[TensorHeatmapPlayback.Frame] {playback?.frames ?? []}
    private var current:TensorHeatmapPlayback.Frame? {guard let index,frames.indices.contains(index) else {return nil};return frames[index]}
    private let timelineRange:ClosedRange<Double>
    init(final:TensorHeatmapSummary,playback:TensorHeatmapPlayback?,accent:Color) {
        self.final=final;self.playback=playback;self.accent=accent
        let values=([final]+(playback?.frames.map(\.summary) ?? [])).flatMap(\.planes).flatMap {$0.minimum+$0.maximum}.compactMap {$0}
        let low=values.min() ?? -1,high=values.max() ?? 1
        let extent=max(abs(low),abs(high),1e-12)
        timelineRange=low<high ? low...high:-extent...extent
    }
    var body:some View {
        VStack(alignment:.leading,spacing:8) {
            if !frames.isEmpty {
                HStack {
                    Button(playing ? "Pause":"Play") {if playing {playing=false} else {if index==nil || index==frames.count-1 {index=0};playing=true}}
                        .buttonStyle(FlatButtonStyle(prominent:playing))
                    Button("Final capture") {playing=false;index=nil}.buttonStyle(FlatButtonStyle(prominent:index==nil))
                    Text(current.map {"Epoch \($0.epoch) · \($0.sourceRows) validation rows"} ?? "Selected model · final capture").font(.caption).foregroundStyle(MLTheme.muted)
                }
                if frames.count>1 {
                    Slider(value:Binding(get:{Double(index ?? frames.count-1)},set:{playing=false;index=Int($0)}),in:0...Double(frames.count-1),step:1)
                        .tint(accent).accessibilityLabel("Captured epoch frame")
                }
            }
            if let measured=current?.summary ?? (final.available ? final:nil) {
                TensorHeatmapView(summary:measured,accent:accent,timelineRange:timelineRange)
            } else {Text(final.reason ?? "Final capture unavailable. Play a retained epoch frame.").font(.caption).foregroundStyle(MLTheme.muted)}
            if let reason=playback?.reason {Text(reason).font(.caption).foregroundStyle(MLTheme.muted)}
        }.task(id:playing) {
            guard playing else {return}
            while !Task.isCancelled && playing {
                do {try await Task.sleep(for:.milliseconds(750))} catch {return}
                guard !Task.isCancelled,playing else {return}
                let next=(index ?? 0)+1
                if next>=frames.count {playing=false;return}
                index=next
            }
        }.onDisappear {playing=false}
    }
}
