import Foundation

/// A finite, reviewed acquisition surface, separate from arbitrary Hub repositories.
public enum OCRPackCatalog {
    public static let id="english-printed-v1"
    public struct Asset:Sendable {
        public let name:String
        public let url:String
        public let bytes:Int
        public let sha256:String
    }
    public static let assets:[Asset]=[
        .init(name:"craft_mlt_25k.zip",url:"https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",bytes:77_251_756,sha256:"8dc6a1c703a89ed56308ef742d26ebd45c656248cbbbda6e7fe60e569f873e65"),
        .init(name:"english_g2.zip",url:"https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip",bytes:14_040_947,sha256:"1b5eaebf1c062de6205560c97ffcfa8dc0e6f413c340e8adc5cfc57e159f61ff")]
    public static var bytes:Int {assets.reduce(0){$0+$1.bytes}}
    public static func asset(_ name:String)->Asset? {assets.first {$0.name==name}}
    public static func allowed(_ url:URL)->Bool {
        guard url.scheme=="https",url.user==nil,url.password==nil,url.port==nil else {return false}
        return assets.contains {$0.url==url.absoluteString} || (url.host=="release-assets.githubusercontent.com" && url.path.hasPrefix("/github-production-release-asset/247266215/"))
    }
}
