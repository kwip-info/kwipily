import SwiftUI
import AVFoundation
import AppKit
import KwipMLCore

/// Decodes once per explicit selection, away from canvas layout and rendering.
struct MediaArtifactPreviewView:View {
    let reference:MediaArtifactReference
    let resolver:FileImageArtifactResolver
    var selectionInterval:ClosedRange<Double>? = nil
    @Environment(\.dismiss) private var dismiss
    @State private var decoded:MediaArtifactReference.Decoded?
    @State private var parentImage:FileImageReference?
    @State private var thumbnail:CGImage?
    @State private var zoom=1.0
    @State private var spectralChannel=0
    @State private var spectralScale=""
    @State private var viewMode="preview"
    @State private var playbackTask:Task<Void,Never>?
    @State private var parent:MediaArtifactReference?
    @State private var showSource=false
    @State private var waveform:[WaveBin]=[]
    @State private var player:AVAudioPlayer?
    @State private var position=0.0
    @State private var failure=""
    @State private var valueOffset=0
    private var active:MediaArtifactReference {showSource ? parent ?? reference:reference}
    private var audio:Bool {active.dtype != "int64" && active.axes.contains("samples") && (active.kind=="audioSamplesRef" || active.axes==["channels","samples"])}
    private var spectral:Bool {active.axes == ["channels","frames","frequency"]}
    private var localInterval:ClosedRange<Double>? {
        guard let selectionInterval else {return nil}
        let offset=Double(active.metadata["sourceStart"]?.integerValue ?? 0)/Double(active.metadata["sourceRate"]?.integerValue ?? 1)
        let lower=max(0,selectionInterval.lowerBound-offset),upper=min(duration,selectionInterval.upperBound-offset)
        return lower<upper ? lower...upper:nil
    }
    private var spectralSummary:String {
        guard case .object(let spec)?=active.metadata["spectral"] else {return ""}
        return "FFT "+String(spec["fftSize"]?.integerValue ?? 0)+" · hop "+String(spec["frameHopSamples"]?.integerValue ?? 0)+" samples · "+(spec["window"]?.stringValue ?? "")+" · "+(spec["representation"]?.stringValue ?? "")
    }
    private var duration:Double {guard let n=active.shape.indices.first(where:{active.axes[$0]=="samples"}),let rate=sampleRate(active),rate>0 else {return 0};return Double(active.shape[n])/Double(rate)}
    var body:some View {
        VStack(alignment:.leading,spacing:14) {
            HStack {
                Label(audio ? "Captured audio":"Prepared tensor",systemImage:audio ? "waveform":"square.stack.3d.up").font(.title3.weight(.semibold))
                Spacer();Button("Close") {stop();dismiss()}.buttonStyle(FlatButtonStyle()).keyboardShortcut(.cancelAction)
            }
            if parent != nil || parentImage != nil {
                FlatChoiceControl(title:"Representation",selection:showSource ? "source":"prepared",options:[.init(id:"source",title:"Source"),.init(id:"prepared",title:"Prepared")]) {value in stop();showSource=value=="source"}
            }
            if showSource,let parentImage {
                ScrollView {FileImagePreviewView(reference:parentImage,resolver:resolver,embedded:true)}
            } else {
            HStack {badge(active.shape.map(String.init).joined(separator:" × "));badge(active.axes.joined(separator:" × "));badge(active.dtype);if let rate=sampleRate(active) {badge(String(rate)+" Hz")}}
            Label("Verified captured artifact · display does not transform model data",systemImage:"checkmark.shield").font(.caption).foregroundStyle(MLTheme.teal)
            if case .string(let convention)?=active.metadata["maskConvention"] {Text(convention=="valid" ? "Mask · 1 = valid "+(active.axes.contains("frames") ? "frame":"sample")+"; 0 = padding":"Mask · 1 = padding / blocked; 0 = valid "+(active.axes.contains("frames") ? "frame":"sample")).font(.caption).foregroundStyle(MLTheme.muted)}
            if decoded != nil {
                let visual=audio || spectral || active.axes.contains("height")
                if visual {FlatChoiceControl(title:"View",selection:viewMode,options:[.init(id:"preview",title:spectral ? "Spectrum":audio ? "Waveform":"Image"),.init(id:"values",title:"Values")]){viewMode=$0}}
                if viewMode != "values" && visual {
                if audio {
                    Canvas {context,size in
                        if showSource,case .integer(let start)?=reference.metadata["sourceStart"],case .integer(let end)?=reference.metadata["sourceEnd"],let axis=active.axes.firstIndex(of:"samples") {
                            let total=Double(active.shape[axis]);let rect=CGRect(x:Double(start)/total*size.width,y:0,width:Double(end-start)/total*size.width,height:size.height)
                            context.fill(Path(rect),with:.color(MLTheme.teal.opacity(0.12)))
                        }
                        if let range=localInterval,duration>0 {
                            context.fill(Path(CGRect(x:range.lowerBound/duration*size.width,y:0,width:(range.upperBound-range.lowerBound)/duration*size.width,height:size.height)),with:.color(MLTheme.teal.opacity(0.18)))
                        }
                        var path=Path()
                        for (i,bin) in waveform.enumerated() {
                            let x=Double(i)*size.width/Double(max(1,waveform.count-1))
                            path.move(to:CGPoint(x:x,y:size.height*(0.5-Double(bin.low)*0.45)))
                            path.addLine(to:CGPoint(x:x,y:size.height*(0.5-Double(bin.high)*0.45)))
                        }
                        context.stroke(path,with:.color(MLTheme.teal),lineWidth:1)
                    }.frame(height:160).background(MLTheme.teal.opacity(0.04)).overlay(Rectangle().stroke(MLTheme.rule))
                    Text(selectionInterval.map {String(format:"Selected segment · %.2f–%.2f s in source recording. Highlight and playback use this interval.",$0.lowerBound,$0.upperBound)} ?? "Waveform · channel 1 · full selected artifact; highlighted interval identifies the prepared window when viewing its source.").font(.caption2).foregroundStyle(MLTheme.muted)
                    HStack {
                        Button("Play") {play()}.buttonStyle(FlatButtonStyle()).disabled(player==nil)
                        Button("Stop") {stop()}.buttonStyle(FlatButtonStyle())
                        Text(String(format:"%.2f s · no autoplay",duration)).font(.caption).foregroundStyle(MLTheme.muted)
                    }
                    Slider(value:$position,in:0...max(duration,0.001)) {editing in if !editing {player?.currentTime=position}}.accessibilityLabel("Audio position in seconds")
                }
                if spectral {
                    FlatChoiceControl(title:"Spectral channel",selection:String(spectralChannel),options:(0..<active.shape[0]).map{.init(id:String($0),title:"Channel "+String($0+1))}){spectralChannel=Int($0) ?? 0}
                    Text("Time → frames · frequency ↑ bins. Complete-channel averages · at most 512 × 256 display cells. Exact data is in Values.").font(.caption2).foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true)
                    Text(spectralSummary).font(.caption2).foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true)
                }
                if !spectralScale.isEmpty {Text(spectralScale).font(.caption2).foregroundStyle(MLTheme.muted)}
                if let thumbnail {imagePreview(thumbnail)}
                }
                if viewMode == "values" || !visual {values}
                if case .integer(let start)?=active.metadata["sourceStart"],case .integer(let end)?=active.metadata["sourceEnd"] {
                    Text("Source sample interval [\(start), \(end)) · valid "+(active.metadata["validFrames"] != nil ? "frames \(active.metadata["validFrames"]?.integerValue ?? 0)":"samples \(active.metadata["validSamples"]?.integerValue ?? 0)")).font(.caption).foregroundStyle(MLTheme.muted)
                }
            } else if failure.isEmpty {ProgressView("Verifying captured media…").frame(maxWidth:.infinity,maxHeight:.infinity)}
            if !failure.isEmpty {Text(failure).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
            Text("Artifact · "+active.sha256).font(.system(size:10,design:.monospaced)).foregroundStyle(MLTheme.muted).textSelection(.enabled)
            }
        }.padding(20).frame(minWidth:640,idealWidth:800,minHeight:560,idealHeight:680)
            .task(id:reference.sha256) {
                do {let resolver=self.resolver,reference=self.reference;let worker=Task.detached(priority:.utility) {(try resolver.parent(reference),try resolver.parentImage(reference))};let pair=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()});try Task.checkCancellation();parent=pair.0;parentImage=pair.1} catch {failure=error.localizedDescription}
            }
            .task(id:active.sha256+":"+String(spectralChannel)) {await load()}
            .onDisappear {stop();player=nil;decoded=nil;waveform=[];thumbnail=nil}
    }
    private func imagePreview(_ thumbnail:CGImage)->some View {
        VStack(alignment:.leading,spacing:8) {
            HStack {Text("Prepared display preview").font(.caption.weight(.semibold));Spacer();Button("Fit") {zoom=1}.buttonStyle(FlatButtonStyle());Text(String(format:"%.0f%%",zoom*100)).font(.caption)}
            Slider(value:$zoom,in:0.5...4).accessibilityLabel("Prepared image display zoom")
            GeometryReader {geometry in
                let fit=min(geometry.size.width/Double(thumbnail.width),geometry.size.height/Double(thumbnail.height))
                ScrollView([.horizontal,.vertical]) {
                    Image(decorative:thumbnail,scale:1).resizable().interpolation(.none)
                        .frame(width:Double(thumbnail.width)*fit*zoom,height:Double(thumbnail.height)*fit*zoom)
                        .frame(minWidth:geometry.size.width,minHeight:geometry.size.height)
                }
            }.frame(height:spectral ? 150:180).overlay(Rectangle().stroke(MLTheme.rule))
            Text(spectral ? "Complete-channel mean aggregation · display colors do not change captured tensor values.":"Nearest-sampled display · at most 512 × 512 pixels. Unsigned pixels use their full range; float32 display uses 0…1 and clips other values. Exact captured values below are unchanged.").font(.caption2).foregroundStyle(MLTheme.muted)
        }
    }
    private var values:some View {
        VStack(alignment:.leading,spacing:8) {
            HStack {
                Text("Exact values · row-major coordinates").font(.caption.weight(.semibold));Spacer()
                Button("Previous") {valueOffset=max(0,valueOffset-64)}.buttonStyle(FlatButtonStyle()).disabled(valueOffset==0)
                Button("Next") {valueOffset+=64}.buttonStyle(FlatButtonStyle()).disabled(valueOffset+64>=active.shape.reduce(1,*))
            }
            if let decoded {
                ScrollView {
                    LazyVStack(alignment:.leading,spacing:5) {
                        ForEach(valueOffset..<min(valueOffset+64,active.shape.reduce(1,*)),id:\.self) {i in
                            HStack {Text(coordinate(i));Spacer();Text(Self.number(decoded.samples,index:i,dtype:active.dtype))}.font(.system(size:11,design:.monospaced)).textSelection(.enabled)
                        }
                    }.padding(10)
                }.frame(minHeight:100,maxHeight:.infinity).overlay(Rectangle().stroke(MLTheme.rule))
            }
        }
    }
    private func badge(_ text:String)->some View {Text(text).font(.system(size:11,design:.monospaced)).padding(6).background(MLTheme.teal.opacity(0.08)).overlay(Rectangle().stroke(MLTheme.rule))}
    private func coordinate(_ linear:Int)->String {var n=linear;var coords=[Int]();for d in active.shape.reversed() {coords.insert(n%d,at:0);n/=d};return "["+coords.map(String.init).joined(separator:", ")+"]"}
    private func sampleRate(_ ref:MediaArtifactReference)->Int? {ref.metadata["sampleRate"]?.integerValue}
    private func stop(){playbackTask?.cancel();playbackTask=nil;player?.stop();player?.currentTime=localInterval?.lowerBound ?? 0;position=localInterval?.lowerBound ?? 0}
    private func play(){
        playbackTask?.cancel()
        if let range=localInterval,!range.contains(position) {position=range.lowerBound}
        player?.currentTime=position;player?.play()
        playbackTask=Task {@MainActor in
            while !Task.isCancelled,let player,player.isPlaying {
                position=player.currentTime
                if let range=localInterval,position>=range.upperBound {player.stop();position=range.lowerBound;break}
                try? await Task.sleep(for:.milliseconds(100))
            }
        }
    }
    @MainActor private func load()async {
        stop();player=nil;decoded=nil;waveform=[];thumbnail=nil;spectralScale="";zoom=1;valueOffset=0;failure=""
        let selected=active,resolver=self.resolver,channel=spectralChannel
        do {
            let worker=Task.detached(priority:.utility) {()->(MediaArtifactReference.Decoded,[WaveBin],Data?,CGImage?,String) in
                let decoded=try resolver.read(selected)
                let origin=selected.kind=="audioSamplesRef" ? selected:try resolver.parent(selected)
                let spectrum=selected.axes == ["channels","frames","frequency"] ? try SpectralPreview.make(decoded,channel:channel):nil
                return (decoded,try Self.wave(decoded,source:origin),try Self.playback(decoded,source:origin),try spectrum?.image ?? Self.image(decoded,channel:channel),spectrum?.scale ?? "")
            }
            let result=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
            guard !Task.isCancelled,active==selected else {return}
            decoded=result.0;waveform=result.1;thumbnail=result.3;spectralScale=result.4
            if let bytes=result.2 {player=try AVAudioPlayer(data:bytes);player?.prepareToPlay()}
        } catch {if !Task.isCancelled {failure=error.localizedDescription}}
    }
    struct WaveBin:Sendable {let low:Float;let high:Float}
    nonisolated private static func scalar(_ data:Data,index:Int,dtype:String)->Double {
        data.withUnsafeBytes {raw in switch dtype {
        case "uint8":return Double(raw[index])
        case "uint16":return Double(UInt16(littleEndian:raw.loadUnaligned(fromByteOffset:index*2,as:UInt16.self)))
        case "int16":return Double(Int16(littleEndian:raw.loadUnaligned(fromByteOffset:index*2,as:Int16.self)))
        case "int32":return Double(Int32(littleEndian:raw.loadUnaligned(fromByteOffset:index*4,as:Int32.self)))
        case "int64":return Double(Int64(littleEndian:raw.loadUnaligned(fromByteOffset:index*8,as:Int64.self)))
        default:return Double(Float(bitPattern:UInt32(littleEndian:raw.loadUnaligned(fromByteOffset:index*4,as:UInt32.self))))
        }}
    }
    nonisolated private static func number(_ data:Data,index:Int,dtype:String)->String {
        if dtype=="int64" {return data.withUnsafeBytes {String(Int64(littleEndian:$0.loadUnaligned(fromByteOffset:index*8,as:Int64.self)))}}
        let value=scalar(data,index:index,dtype:dtype);return dtype=="float32" ? String(format:"%.9g",value):String(format:"%.0f",value)
    }
    nonisolated static func amplitude(_ data:Data,index:Int,ref:MediaArtifactReference,source:MediaArtifactReference?=nil)->Double {
        let value=scalar(data,index:index,dtype:ref.dtype)
        if ref.dtype != "float32",let bits=(source ?? ref).metadata["sourceBits"]?.integerValue {return bits==8 ? (value-128)/128:value/pow(2,Double(bits-1))}
        return value
    }
    nonisolated private static func wave(_ decoded:MediaArtifactReference.Decoded,source:MediaArtifactReference?=nil)throws->[WaveBin] {
        let ref=decoded.reference;guard let axis=ref.axes.firstIndex(of:"samples"),ref.shape.count<=2,ref.dtype != "int64" else {return []}
        let frames=ref.shape[axis],channels=ref.shape.count==2 ? ref.shape[1-axis]:1,bins=min(1024,frames)
        var out:[WaveBin]=[]
        for bin in 0..<bins {
            try Task.checkCancellation();var low=0.0,high=0.0
            for frame in (bin*frames/bins)..<((bin+1)*frames/bins) {
                let index=axis==0 ? frame*channels:frame
                let value=max(-1,min(1,amplitude(decoded.samples,index:index,ref:ref,source:source)))
                low=min(low,value);high=max(high,value)
            }
            out.append(.init(low:Float(low),high:Float(high)))
        }
        return out
    }
    nonisolated private static func image(_ decoded:MediaArtifactReference.Decoded,channel:Int=0)throws->CGImage? {
        let ref=decoded.reference
        guard ref.shape.count==3,let y=ref.axes.firstIndex(of:"height"),let x=ref.axes.firstIndex(of:"width"),let c=ref.axes.firstIndex(of:"channels"),["uint8","uint16","float32"].contains(ref.dtype) else {return nil}
        let height=ref.shape[y],width=ref.shape[x],channels=ref.shape[c]
        guard (1...4).contains(channels) else {return nil}
        let w=min(width,512),h=min(height,512),denominator=ref.dtype=="uint8" ? 255.0:ref.dtype=="uint16" ? 65535.0:1.0
        var rgba=[UInt8](repeating:0,count:w*h*4)
        func sample(_ row:Int,_ column:Int,_ channel:Int)->Double {
            var coordinates=[Int](repeating:0,count:3);coordinates[y]=row;coordinates[x]=column;coordinates[c]=channel
            let index=(coordinates[0]*ref.shape[1]+coordinates[1])*ref.shape[2]+coordinates[2]
            return max(0,min(1,scalar(decoded.samples,index:index,dtype:ref.dtype)/denominator))
        }
        for row in 0..<h {
            try Task.checkCancellation()
            for column in 0..<w {
                let sy=row*height/h,sx=column*width/w,hasAlpha=channels==2 || channels==4
                let alpha=hasAlpha ? sample(sy,sx,channels-1):1
                let offset=(row*w+column)*4
                for component in 0..<3 {rgba[offset+component]=UInt8((sample(sy,sx,channels<=2 ? 0:component)*alpha*255).rounded())}
                rgba[offset+3]=UInt8((alpha*255).rounded())
            }
        }
        guard let provider=CGDataProvider(data:Data(rgba) as CFData) else {return nil}
        return CGImage(width:w,height:h,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:w*4,space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.premultipliedLast.rawValue),provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent)
    }
    nonisolated private static func playback(_ decoded:MediaArtifactReference.Decoded,source:MediaArtifactReference?=nil)throws->Data? {
        let ref=decoded.reference
        guard let axis=ref.axes.firstIndex(of:"samples"),ref.shape.count==2,ref.dtype != "int64",let rate=ref.metadata["sampleRate"]?.integerValue else {return nil}
        let frames=ref.shape[axis],channels=ref.shape[1-axis]
        guard channels<=2,frames*channels<=12_000_000 else {return nil}
        var payload=Data(capacity:frames*channels*4)
        for frame in 0..<frames {
            if frame%4096==0 {try Task.checkCancellation()}
            for channel in 0..<channels {
                let i=axis==0 ? frame*channels+channel:channel*frames+frame
                var sample=Float(max(-1,min(1,amplitude(decoded.samples,index:i,ref:ref,source:source)))).bitPattern.littleEndian
                withUnsafeBytes(of:&sample) {payload.append(contentsOf:$0)}
            }
        }
        func little<T:FixedWidthInteger>(_ n:T)->Data {var v=n.littleEndian;return withUnsafeBytes(of:&v){Data($0)}}
        var wav=Data("RIFF".utf8);wav.append(little(UInt32(36+payload.count)));wav.append(Data("WAVEfmt ".utf8));wav.append(little(UInt32(16)));wav.append(little(UInt16(3)));wav.append(little(UInt16(channels)));wav.append(little(UInt32(rate)));wav.append(little(UInt32(rate*channels*4)));wav.append(little(UInt16(channels*4)));wav.append(little(UInt16(32)));wav.append(Data("data".utf8));wav.append(little(UInt32(payload.count)));wav.append(payload);return wav
    }
}
