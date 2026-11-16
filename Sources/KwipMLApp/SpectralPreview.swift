import Foundation
import CoreGraphics
import KwipMLCore

/// Complete-channel aggregation computed only for an explicitly opened artifact.
enum SpectralPreview {
    struct Raster {let width:Int;let height:Int;let means:[Double];let minimum:Double;let maximum:Double}
    struct Preview {let image:CGImage;let scale:String}
    nonisolated static func raster(samples:Data,shape:[Int],channel:Int)throws->Raster {
        var count=1
        for dimension in shape {guard dimension>0,dimension<=12_497_500/count else {throw WorkflowArchiveError("Spectral display exceeds its decoded byte allowance.")};count*=dimension}
        guard shape.count==3,samples.count==count*4,(0..<shape[0]).contains(channel) else {throw WorkflowArchiveError("Spectral display requires a bounded captured channels × frames × frequency tensor.")}
        let frames=shape[1],bins=shape[2],w=min(512,frames),h=min(256,bins),base=channel*frames*bins
        var low=Double.infinity,high = -Double.infinity,means=[Double](repeating:0,count:w*h)
        try samples.withUnsafeBytes {raw in
            func value(_ i:Int)->Double {Double(Float(bitPattern:UInt32(littleEndian:raw.loadUnaligned(fromByteOffset:(base+i)*4,as:UInt32.self))))}
            for i in 0..<(frames*bins) {if i%4096==0 {try Task.checkCancellation()};let v=value(i);guard v.isFinite else {throw WorkflowArchiveError("Spectral display values must be finite.")};low=min(low,v);high=max(high,v)}
            for y in 0..<h {
                try Task.checkCancellation();let a=(h-1-y)*bins/h,b=(h-y)*bins/h
                for x in 0..<w {
                    let c=x*frames/w,d=(x+1)*frames/w;var sum=0.0
                    for frame in c..<d {for bin in a..<b {sum+=value(frame*bins+bin)}}
                    means[y*w+x]=sum/Double((b-a)*(d-c))
                }
            }
        }
        return .init(width:w,height:h,means:means,minimum:low,maximum:high)
    }
    nonisolated static func make(_ decoded:MediaArtifactReference.Decoded,channel:Int)throws->Preview {
        let ref=decoded.reference
        guard ref.axes == ["channels","frames","frequency"],ref.dtype=="float32" else {throw WorkflowArchiveError("Spectral display needs explicit frequency/frame axes.")}
        let raster=try raster(samples:decoded.samples,shape:ref.shape,channel:channel)
        var rgba=[UInt8](repeating:255,count:raster.width*raster.height*4)
        for (i,v) in raster.means.enumerated() {
            let t=raster.maximum>raster.minimum ? max(0,min(1,(v-raster.minimum)/(raster.maximum-raster.minimum))):0
            rgba[i*4]=UInt8(30+200*t);rgba[i*4+1]=UInt8(20+210*t);rgba[i*4+2]=UInt8(70+100*(1-t))
        }
        guard let provider=CGDataProvider(data:Data(rgba) as CFData),let image=CGImage(width:raster.width,height:raster.height,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:raster.width*4,space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.premultipliedLast.rawValue),provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent) else {throw WorkflowArchiveError("Spectral display image could not be created.")}
        return .init(image:image,scale:String(format:"Color range · %.4g to %.4g · complete selected channel",raster.minimum,raster.maximum))
    }
}
