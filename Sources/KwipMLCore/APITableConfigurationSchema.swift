import Foundation

/// Fixed diagnostic vocabulary: no submitted URLs, values, paths or credentials.
public enum APIConfigurationIssue:String {
    case inputEnvelope,outputEnvelope,fields,url,requestFields,mapping
    public var candidateFields:[String] {
        switch self {
        case .inputEnvelope:return ["format","options"]
        case .outputEnvelope:return ["destination","api"]
        case .fields:return ["options","api"]
        case .url:return ["url"]
        case .requestFields:return ["headers","query"]
        case .mapping:return ["mapping"]
        }
    }
    public var guidance:String {
        switch self {
        case .inputEnvelope:return "API Table Input replaces the entire configuration with format:api and an options object. Remove CSV-specific top-level fields. Consult catalog_get configurationVariants."
        case .outputEnvelope:return "API Table Output requires destination:api and an api object with direction:output and a supported write method. Keep the advertised Table Output envelope; private connections are not portable settings."
        case .fields:return "API options contain an unsupported field. Use the advertised schema; private credentials and connection grants are not portable settings."
        case .url:return "Use an HTTPS endpoint without embedded credentials, query or fragment. Place non-secret query fields in query; private authentication stays in the app."
        case .requestFields:return "Headers and query accept bounded non-secret string fields only. Credentials belong in the app's private connection profile."
        case .mapping:return "Provide at least one mapping row with source, target and type. Input source is a JSON Pointer; output target is a JSON Pointer. Names must be unique."
        }
    }
}

extension APITableSettings {
    /// Discovery shares validation's choices, limits and defaults. Cross-field
    /// constraints remain enforced by normalized(_:), not inferred from schema.
    public static func configurationSchema(direction:String)->JSONValue {
        func string(_ maximum:Int=128)->JSONValue {.object(["type":.string("string"),"maxLength":.integer(Int64(maximum))])}
        func object(_ properties:[String:JSONValue],required:[String]=[])->JSONValue {.object(["type":.string("object"),"properties":.object(properties),"required":.array(required.map(JSONValue.string)),"additionalProperties":false])}
        var properties:[String:JSONValue]=[:]
        for (key,low,high) in limits {properties[key] = .object(["type":.string("integer"),"minimum":.integer(Int64(low)),"maximum":.integer(Int64(high))])}
        for (key,values) in choices {properties[key] = .object(["type":.string("string"),"enum":.array((key=="direction" ? [direction]:values).map(JSONValue.string))])}
        properties["method"] = .object(["type":.string("string"),"enum":.array((direction=="input" ? ["GET"]:["POST","PUT","PATCH","DELETE"]).map(JSONValue.string))])
        properties["url"]=string(2048);properties["allowLoopback"] = .object(["type":.string("boolean")])
        for key in ["recordsPath","cursorPath","nextPath","arrayPath","responsePath"] {properties[key]=string(1024)}
        for key in ["cursorParameter","pageParameter","credentialHeader","idempotencyHeader"] {properties[key]=string()}
        for key in ["headers","query"] {properties[key] = .object(["type":.string("object"),"maxProperties":32,"additionalProperties":string(1024)])}
        properties["mapping"] = .object(["type":.string("array"),"minItems":1,"maxItems":128,"items":object(["source":string(256),"target":string(256),"type":.object(["type":.string("string"),"enum":.array(types.map(JSONValue.string))])],required:["source","target","type"])])
        properties["constants"] = .object(["type":.string("array"),"maxItems":.integer(direction=="input" ? 0:32),"items":object(["target":string(256),"json":string(1024)],required:["target","json"])])
        return object(properties,required:direction=="input" ? ["url","mapping"]:["url","mapping","direction","method"])
    }
    public static func discoveryVariant(direction:String)->JSONValue {
        let input=direction=="input"
        var api=defaults;api["direction"] = .string(direction);api["method"] = .string(input ? "GET":"POST")
        api["url"] = .string(input ? "https://www.gov.uk/bank-holidays.json":"https://api.example.com/events")
        api["recordsPath"] = .string(input ? "/england-and-wales/events":"")
        api["mapping"] = .array([.object(["source":.string(input ? "/title":"title"),"target":.string(input ? "title":"/title"),"type":.string("string")]),.object(["source":.string(input ? "/date":"date"),"target":.string(input ? "date":"/date"),"type":.string("string")])])
        let example:[String:JSONValue]=input ? ["format":.string("api"),"options":.object(api)]:["destination":.string("api"),"api":.object(api),"filename":.string("events.csv"),"separator":.string(","),"columns":.array([]),"overwrite":false]
        return .object(["name":.string(input ? "API Input":"API Output"),"discriminator":.object([input ? "format":"destination":.string("api")]),"settingsField":.string(input ? "options":"api"),"settingsSchema":configurationSchema(direction:direction),"exampleConfiguration":.object(example),"configurationSemantics":.string(input ? "Replace the entire configuration with format and options; do not retain CSV fields.":"Keep the Table Output envelope and place request settings under api."),"authentication":.string("auth selects none, bearer, basic or apiKey. Credentials and matching-origin connection grants must be selected privately in the app, including unauthenticated endpoints. No secret or profile reference belongs in portable configuration."),"mappingSemantics":.string(input ? "recordsPath selects the JSON records array; each mapping source is a JSON Pointer relative to a record, target is an output column.":"Each mapping source names an input column; target is a JSON Pointer in each delivered record. delivery chooses row or batch."),"executionSemantics":.string("Configuration preview does not fetch or write. Native connection grants and runtime validation remain required; Run performs the request. Retries and delivery receipts follow the configured method and idempotency policy.")])
    }
}
