import Foundation
import CryptoKit
import MCP

public struct MCPBrokerClient:Codable,Equatable,Sendable {
    public let id:String
    public let tokenSHA256:String
    public init(id:String,tokenSHA256:String){self.id=id;self.tokenSHA256=tokenSHA256}
}
public struct MCPBrokerSetup:Codable,Sendable {
    public let fixedPort:Int?
    public let clients:[MCPBrokerClient]
    public init(fixedPort:Int?,clients:[MCPBrokerClient]){self.fixedPort=fixedPort;self.clients=clients}
    public func validated()throws->Self {
        guard fixedPort==nil || (1024...65535).contains(fixedPort!),clients.count<=16,
              Set(clients.map(\.id)).count==clients.count,
              clients.allSatisfy({!$0.id.isEmpty && $0.id.utf8.count<=200 && $0.tokenSHA256.count==64 && $0.tokenSHA256.allSatisfy({"0123456789abcdef".contains($0)})}) else {throw HTTPRefusal(status:400)}
        return self
    }
}
public struct HTTPRefusal:Error,Equatable {public let status:Int;public init(status:Int){self.status=status}}
public enum MCPHTTPAdmission {
    public static let maximumBodyBytes=1_048_576
    public static let maximumHeaderBytes=16_384
    public static func tokenDigest(_ token:String)->String {SHA256.hash(data:Data(token.utf8)).map{String(format:"%02x",$0)}.joined()}
    public static func client(for request:HTTPRequest,port:UInt16,clients:[MCPBrokerClient])throws->String {
        guard request.path=="/mcp",["POST","GET","DELETE"].contains(request.method) else {throw HTTPRefusal(status:404)}
        guard request.header("host")=="127.0.0.1:\(port)" || request.header("host")=="localhost:\(port)" else {throw HTTPRefusal(status:403)}
        if let origin=request.header("origin"),!["http://127.0.0.1:\(port)","http://localhost:\(port)"].contains(origin) {throw HTTPRefusal(status:403)}
        guard (request.body?.count ?? 0)<=maximumBodyBytes else {throw HTTPRefusal(status:413)}
        guard let authorization=request.header("authorization"),authorization.hasPrefix("Bearer ") else {throw HTTPRefusal(status:401)}
        let token=String(authorization.dropFirst(7))
        guard (32...128).contains(token.utf8.count),token.allSatisfy({$0.isASCII && ($0.isLetter || $0.isNumber || $0=="-" || $0=="_")}) else {throw HTTPRefusal(status:401)}
        let hash=Array(tokenDigest(token).utf8)
        // Compare hashes without leaking the first mismatching byte. Never retain
        // bearer plaintext in settings, diagnostic records or URLs.
        var selected:String?
        for client in clients {let expected=Array(client.tokenSHA256.utf8);var difference=expected.count ^ hash.count
            for (a,b) in zip(expected,hash){difference |= Int(a ^ b)}
            if difference==0 {selected=client.id}}
        guard let selected else {throw HTTPRefusal(status:401)};return selected
    }
}
/// One bounded HTTP/1.1 request per connection; standard-library frame adapter for
/// the SDK transport. Reject ambiguous framing instead of accepting request smuggling.
public struct MCPHTTPParser {
    public private(set) var bytes=Data()
    private let webhook:Bool
    public init(webhook:Bool=false){self.webhook=webhook}
    public mutating func append(_ next:Data)throws->HTTPRequest? {
        guard bytes.count+next.count<=MCPHTTPAdmission.maximumHeaderBytes+MCPHTTPAdmission.maximumBodyBytes else {throw HTTPRefusal(status:413)}
        bytes.append(next)
        guard let delimiter=bytes.range(of:Data("\r\n\r\n".utf8)) else {
            guard bytes.count<=MCPHTTPAdmission.maximumHeaderBytes else {throw HTTPRefusal(status:431)};return nil
        }
        guard delimiter.lowerBound<=MCPHTTPAdmission.maximumHeaderBytes,
              let header=String(data:bytes[..<delimiter.lowerBound],encoding:.utf8) else {throw HTTPRefusal(status:400)}
        let lines=header.components(separatedBy:"\r\n")
        let start=lines[0].split(separator:" ",omittingEmptySubsequences:false)
        guard start.count==3,start[2]=="HTTP/1.1",["POST","GET","DELETE"].contains(String(start[0])),(webhook ? (start[0]=="POST" && start[1].hasPrefix("/hooks/") && UUID(uuidString:String(start[1].dropFirst(7))) != nil):start[1]=="/mcp") else {throw HTTPRefusal(status:400)}
        var headers:[String:String]=[:]
        for line in lines.dropFirst(){guard let colon=line.firstIndex(of:":"),colon != line.startIndex else {throw HTTPRefusal(status:400)}
            let name=String(line[..<colon]).lowercased();let value=String(line[line.index(after:colon)...]).trimmingCharacters(in:.whitespaces)
            guard name.allSatisfy({$0.isASCII && ($0.isLetter || $0.isNumber || $0=="-")}),headers[name]==nil,
                  !value.unicodeScalars.contains(where:{$0.value<32 || $0.value==127}) else {throw HTTPRefusal(status:400)}
            headers[name]=value
        }
        guard headers["transfer-encoding"]==nil,headers["expect"]==nil else {throw HTTPRefusal(status:400)}
        let rawLength=headers["content-length"] ?? "0"
        guard let length=Int(rawLength),String(length)==rawLength,(0...MCPHTTPAdmission.maximumBodyBytes).contains(length),
              start[0] != "POST" || headers["content-length"] != nil else {throw HTTPRefusal(status:400)}
        let expected=delimiter.upperBound+length
        guard bytes.count<=expected else {throw HTTPRefusal(status:400)}
        guard bytes.count==expected else {return nil}
        return HTTPRequest(method:String(start[0]),headers:headers,body:Data(bytes[delimiter.upperBound...]),path:String(start[1]))
    }
}
