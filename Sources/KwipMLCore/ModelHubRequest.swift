import Foundation

/// A finite Hugging Face read, not an arbitrary URL/network capability.
public struct ModelHubRequest:Codable,Equatable,Sendable {
    public enum Action:String,Codable,Sendable {case resolve,file,ocrFile}
    public let action:Action
    public let repository:String
    public let revision:String
    public var file:String?=nil
    public var offset:Int64?=nil
    public var total:Int64?=nil
    public var bearer:String?=nil
    public init(action:Action,repository:String,revision:String,file:String?=nil,offset:Int64?=nil,total:Int64?=nil,bearer:String?=nil) {self.action=action;self.repository=repository;self.revision=revision;self.file=file;self.offset=offset;self.total=total;self.bearer=bearer}
    public static let chunkBytes:Int64=1_048_576
    public var maximumResponseBytes:Int {action == .resolve ? 2_000_000:Int(min(Self.chunkBytes,(total ?? 0)-(offset ?? 0)))}
    public func prepared()throws->URLRequest {
        guard LoadModelConfiguration.validRepository(repository),LoadModelConfiguration.validRevision(revision) else {throw DataFlowContractError("Invalid model repository/revision.")}
        if let bearer {guard bearer.hasPrefix("hf_"),bearer.utf8.count<=4096,bearer.unicodeScalars.allSatisfy({$0.value>=33 && $0.value<=126}) else {throw DataFlowContractError("Invalid private Hub credential.")}}
        let url:URL
        switch action {
        case .ocrFile:
            guard repository=="JaidedAI/EasyOCR",revision==OCRPackCatalog.id,bearer==nil,let file,let asset=OCRPackCatalog.asset(file),let offset,let total,total==Int64(asset.bytes),offset>=0,offset<total,offset%Self.chunkBytes==0 else {throw DataFlowContractError("OCR download requires one exact reviewed pack asset and bounded range.")}
            url=URL(string:asset.url)!

        case .resolve:
            guard file==nil,offset==nil,total==nil else {throw DataFlowContractError("Revision resolution does not accept files/ranges.")}
            url=URL(string:"https://huggingface.co/api/models/\(repository)/revision/\(revision)?blobs=true")!
        case .file:
            guard revision.count==40,revision.allSatisfy({$0.isHexDigit && !$0.isUppercase}),let file,Self.allowedFile(file),let offset,let total,total>0,total<=2_000_000_000,offset>=0,offset<total,offset%Self.chunkBytes==0 else {throw DataFlowContractError("Model download needs an immutable revision and a bounded file range.")}
            url=URL(string:"https://huggingface.co/\(repository)/resolve/\(revision)/\(file)")!
        }
        var request=URLRequest(url:url,timeoutInterval:30);request.httpShouldHandleCookies=false
        request.setValue("identity",forHTTPHeaderField:"Accept-Encoding")
        if let bearer {request.setValue("Bearer "+bearer,forHTTPHeaderField:"Authorization")}
        if action == .file || action == .ocrFile {let start=offset!;request.setValue("bytes=\(start)-\(start+Int64(maximumResponseBytes)-1)",forHTTPHeaderField:"Range")}
        return request
    }
    public static let nestedModelFiles:Set<String>=["encoder/config.json","tokenizer/tokenizer.json","tokenizer/tokenizer_config.json","tokenizer/special_tokens_map.json","tokenizer/added_tokens.json"]
    public static func allowedAssetFile(_ path:String)->Bool {nestedModelFiles.contains(path) || (!path.contains("/") && allowedFile(path))}
    public static func allowedFile(_ path:String)->Bool {
        let parts=path.split(separator:"/",omittingEmptySubsequences:false)
        guard (1...3).contains(parts.count),parts.allSatisfy({LoadModelConfiguration.validSegment(String($0))}),let name=parts.last else {return false}
        if parts.count==3 {return nestedModelFiles.contains(parts.suffix(2).joined(separator:"/"))}
        let fixed:Set<String>=["rl_agent_config.json","kwip-decisions.json","kwip-training.json","preprocessor_config.json","normalizer.json","config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","generation_config.json","added_tokens.json","vocab.json","merges.txt","tokenizer.model","model.safetensors.index.json","README.md","LICENSE","LICENSE.txt","NOTICE","NOTICE.txt"]
        return fixed.contains(String(name)) || String(name).range(of:#"^model(?:-\d{5}-of-\d{5})?\.safetensors$"#,options:.regularExpression) != nil
    }
    public static func allowedRedirect(_ url:URL)->Bool {
        guard url.scheme=="https",url.user==nil,url.password==nil,url.port==nil,let host=url.host?.lowercased() else {return false}
        return host=="huggingface.co" || host=="us.aws.cdn.hf.co" || host=="us.gcp.cdn.hf.co" || host=="cdn-lfs.huggingface.co" || host=="cdn-lfs-us-1.hf.co" || host=="cdn-lfs-eu-1.hf.co" || host=="cas-bridge.xethub.hf.co"
    }
}
