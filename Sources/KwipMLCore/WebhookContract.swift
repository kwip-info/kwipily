import Foundation
import CryptoKit

public struct WebhookRefusal:Error,LocalizedError,Sendable {
    public let status:Int
    public let code:String
    public init(_ status:Int,_ code:String){self.status=status;self.code=code}
    public var errorDescription:String? {"Webhook request refused: "+code+". Review the endpoint, authentication and payload settings."}
}
public enum WebhookAuthentication {
    public enum Method:String,Codable,CaseIterable,Sendable {case signed,bearer}
    public static func signature(secret:String,timestamp:String,deliveryID:String,body:Data)->String {
        var message=Data((timestamp+"."+deliveryID+".").utf8);message.append(body)
        return HMAC<SHA256>.authenticationCode(for:message,using:SymmetricKey(data:Data(secret.utf8))).map{String(format:"%02x",$0)}.joined()
    }
    private static func equal(_ a:String,_ b:String)->Bool {
        let x=Array(SHA256.hash(data:Data(a.utf8))),y=Array(SHA256.hash(data:Data(b.utf8)))
        return zip(x,y).reduce(UInt8(0)){$0 | ($1.0 ^ $1.1)}==0
    }
    public static func deliveryID(_ headers:[String:String])throws->String {
        guard let id=headers["x-kwip-delivery-id"],(1...128).contains(id.utf8.count),id.utf8.allSatisfy({(48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || [45,95,46].contains($0)}) else{throw WebhookRefusal(400,"delivery_id_required")};return id
    }
    public static func verify(method:Method,secret:String,headers:[String:String],body:Data,now:Date=Date())throws {
        let id=try deliveryID(headers)
        guard (32...128).contains(secret.utf8.count) else{throw WebhookRefusal(503,"credential_unavailable")}
        switch method {
        case .bearer:guard let received=headers["authorization"],received.utf8.count<=160,equal(received,"Bearer "+secret) else{throw WebhookRefusal(401,"authentication_failed")}
        case .signed:
            guard let stamp=headers["x-kwip-timestamp"],let seconds=Int64(stamp),String(seconds)==stamp,abs(now.timeIntervalSince1970-Double(seconds))<=300,
                  let received=headers["x-kwip-signature"],received.count==64,equal(received,signature(secret:secret,timestamp:stamp,deliveryID:id,body:body)) else{throw WebhookRefusal(401,"signature_or_timestamp_invalid")}
        }
    }
}
public struct WebhookPayload:Codable,Equatable,Sendable {
    public enum ValueType:String,Codable,CaseIterable,Sendable {case string,int64,float64,bool}
    public struct Column:Codable,Equatable,Identifiable,Sendable {
        public var field:String;public var column:String;public var type:ValueType;public var included:Bool
        public var id:String {field}
        public init(field:String,column:String,type:ValueType,included:Bool=true){self.field=field;self.column=column;self.type=type;self.included=included}
    }
    public var tableNodeID:String
    public var recordsPath:String
    public var columns:[Column]
    public init(tableNodeID:String="",recordsPath:String="",columns:[Column]=[]){self.tableNodeID=tableNodeID;self.recordsPath=recordsPath;self.columns=columns}
    public func validate()throws {
        let selected=columns.filter(\.included)
        guard (1...128).contains(tableNodeID.count),recordsPath.count<=256,columns.count<=128,!selected.isEmpty,
              Set(selected.map(\.column)).count==selected.count,
              columns.allSatisfy({!$0.field.isEmpty && $0.field.count<=128 && !$0.column.isEmpty && $0.column.count<=128 && !$0.column.contains("\0")}),
              recordsPath.isEmpty || recordsPath.split(separator:".",omittingEmptySubsequences:false).allSatisfy({!$0.isEmpty}) else {throw WebhookRefusal(422,"invalid_column_mapping")}
    }
    public func decode(_ body:Data)throws->(data:Data,rows:Int) {
        try validate();guard body.count<=1_048_576 else{throw WebhookRefusal(413,"payload_too_large")}
        let root:JSONValue
        do {root=try JSONDecoder().decode(JSONValue.self,from:body)}catch{throw WebhookRefusal(422,"invalid_json")}
        var value=root
        if !recordsPath.isEmpty {for part in recordsPath.split(separator:".") {guard case .object(let object)=value,let next=object[String(part)] else{throw WebhookRefusal(422,"records_path_missing")};value=next}}
        guard case .array(let records)=value,records.count<=10000 else{throw WebhookRefusal(422,"records_array_required")}
        let selected=columns.filter(\.included)
        guard records.count*selected.count<=100000 else{throw WebhookRefusal(413,"table_cell_limit")}
        var rows:[JSONValue]=[]
        for record in records {
            guard case .object(let fields)=record else{throw WebhookRefusal(422,"record_object_required")}
            var row:[String:JSONValue]=[:]
            for column in selected {
                guard let field=fields[column.field],field != .null else{throw WebhookRefusal(422,"required_field_missing")}
                switch (column.type,field) {
                case (.string,.string),(.bool,.bool),(.int64,.integer):break
                case (.float64,.number(let d)):guard d.isFinite else{throw WebhookRefusal(422,"nonfinite_value")}
                case (.float64,.integer(let n)):guard abs(Double(n))<=9007199254740992 else{throw WebhookRefusal(422,"lossy_numeric_conversion")}
                default:throw WebhookRefusal(422,"column_type_mismatch")
                }
                row[column.column]=field
            }
            rows.append(.object(row))
        }
        let bytes=try JSONEncoder().encode(JSONValue.array(rows))
        guard bytes.count<=1_048_576 else{throw WebhookRefusal(413,"mapped_payload_too_large")}
        return (bytes,rows.count)
    }
}
/// Only this bounded envelope crosses the app/helper connection. Headers are never persisted.
public struct WebhookHTTPRequest:Codable,Sendable {
    public var path:String;public var headers:[String:String];public var body:Data
    public init(path:String,headers:[String:String],body:Data){self.path=path;self.headers=headers;self.body=body}
}
public struct WebhookHTTPResponse:Codable,Sendable {
    public var status:Int;public var code:String;public var delivery:String?;public var job:String?
    public init(status:Int,code:String,delivery:String?=nil,job:String?=nil){self.status=status;self.code=code;self.delivery=delivery;self.job=job}
}
