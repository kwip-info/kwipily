import Foundation
import KwipMLCore

/// Public option inventory matches the bounded Python writers. Hidden conditional
/// options retain their draft value; switching formats creates a clean option set.
enum TableOutputConfiguration {
    struct Field: Identifiable {
        enum Kind { case text, integer, optionalInteger, boolean, choices([(String,String)]) }
        let id: String
        let title: String
        let kind: Kind
        let initial: JSONValue
    }
    static let formats = [("csv","CSV"),("tsv","TSV"),("parquet","Parquet"),("json","JSON"),("ndjson","NDJSON"),("xlsx","Excel (.xlsx)")]
    static func fields(_ format: String) -> [Field] {
        func text(_ id:String,_ title:String,_ value:String)->Field { Field(id:id,title:title,kind:.text,initial:.string(value)) }
        func flag(_ id:String,_ title:String,_ value:Bool)->Field {Field(id:id,title:title,kind:.boolean,initial:.bool(value))}
        func choice(_ id:String,_ title:String,_ value:String,_ choices:[(String,String)])->Field {Field(id:id,title:title,kind:.choices(choices),initial:.string(value))}
        switch format {
        case "csv","tsv": return [
            flag("header","Include header",true),
            choice("encoding","Encoding","utf-8",[("utf-8","UTF-8"),("utf-16-le","UTF-16 little endian"),("utf-16-be","UTF-16 big endian")]),flag("bom","Include byte-order mark",false),
            choice("newline","Line endings","LF",[("LF","LF"),("CRLF","CRLF")]),
            choice("quoting","Quoting","minimal",[("minimal","When required"),("all","All fields"),("none","None · escape required")]),text("quoteChar","Quote character","\""),
            choice("escaping","Embedded quote handling","double",[("double","Double quote"),("escape","Escape character"),("none","Stop if escaping is required")]),text("escapeChar","Escape character","\\"),
            text("nullText","Missing-value text",""),choice("numericFormat","Number formatting","roundTrip",[("roundTrip","Full round trip"),("custom","Custom · may reduce precision")]),text("numericPattern","Numeric pattern",".6g"),
            choice("temporalFormat","Date/time formatting","iso",[("iso","ISO · preserve microseconds"),("pattern","Explicit pattern")]),text("temporalPattern","Temporal pattern","%Y-%m-%dT%H:%M:%S.%f%z"),
            choice("spreadsheetText","Spreadsheet-like text","preserve",[("preserve","Preserve exact text"),("apostrophe","Prefix apostrophe"),("stop","Stop run")])]
        case "json","ndjson":
            var result:[Field]=[]
            if format == "json" {result += [choice("shape","JSON shape","records",[("records","Records"),("rowsSchema","Rows plus schema")]),flag("pretty","Pretty print",false)]}
            result += [flag("ascii","Escape non-ASCII characters",false),choice("nullPolicy","Missing values","null",[("null","JSON null"),("omit","Omit missing record keys")]),choice("temporal","Date/time representation","iso",[("iso","ISO text · microseconds"),("integer","Canonical integer units")]),choice("nonfinite","Nonfinite numbers","stop",[("stop","Stop run"),("null","Use JSON null")])]
            return result
        case "parquet":return [choice("compression","Compression","snappy",[("none","None"),("snappy","Snappy"),("gzip","Gzip"),("brotli","Brotli"),("zstd","Zstandard"),("lz4","LZ4")]),Field(id:"compressionLevel",title:"Compression level · blank uses codec default",kind:.optionalInteger,initial:.null),Field(id:"rowGroupSize",title:"Rows per row group",kind:.integer,initial:.integer(65536)),Field(id:"dataPageSize",title:"Target data-page bytes",kind:.integer,initial:.integer(1048576)),flag("statistics","Write column statistics",true)]
        case "xlsx":return [text("sheet","Sheet name","Table"),text("startCell","Start cell","A1"),flag("header","Include header",true),flag("freeze","Freeze rows/columns before data",false),flag("autofilter","Enable header autofilter",false),text("numberFormat","Number format","General"),text("dateFormat","Date/time format","yyyy-mm-dd hh:mm:ss.000"),choice("precisionPolicy","Values Excel cannot preserve exactly","stop",[("stop","Stop run"),("text","Write exact text")])]
        default:return []
        }
    }
    static func defaults(_ format:String)->[String:JSONValue] {Dictionary(uniqueKeysWithValues:fields(format).map {($0.id,$0.initial)})}
    static func visible(_ field:Field,format:String,options:[String:JSONValue])->Bool {
        switch field.id {
        case "escapeChar": return options["escaping"] == .string("escape")
        case "quoteChar": return options["quoting"] != .string("none")
        case "escaping": return options["quoting"] != .string("none")
        case "numericPattern":return options["numericFormat"] == .string("custom")
        case "temporalPattern":return options["temporalFormat"] == .string("pattern")
        case "compressionLevel":return ["gzip","brotli","zstd"].contains(string(options["compression"]))
        case "nullPolicy":return options["shape"] != .string("rowsSchema")
        case "autofilter":return options["header"] != .bool(false)
        default:return true
        }
    }
    static func validate(_ config:[String:JSONValue],definition:DataFlowOperatorDefinition,schema:[DataColumnSchema])throws->[String:JSONValue] {
        _ = try DataEngineeringConfiguration.validate(config,definition:definition,schema:schema)
        if APITableSettings.isAPI(config) {
            guard case .object(let raw)=config["api"] else {throw DataFlowContractError("Configure the API destination.")}
            let c=try APITableSettings.normalized(raw)
            guard c["direction"] == .string("output") else {throw DataFlowContractError("API destination must send output.")}
            if case .array(let rows)=c["mapping"], !schema.isEmpty {let names=Set(schema.map(\.name));guard rows.allSatisfy({if case .object(let r)=$0 {return r["source"]?.stringValue.map(names.contains) ?? false};return false}) else {throw DataFlowContractError("Choose columns from the connected table.")}}
            return config
        }
        if DatabaseOutputSettings.isDatabase(config) {
            guard case .object(let db)=config["database"] else {throw DataFlowContractError("Configure the Database destination.")}
            let normalized = try DatabaseOutputSettings.normalized(db)
            if case .array(let mapping)=normalized["mapping"], !schema.isEmpty {
                let available=Set(schema.map(\.name))
                guard mapping.allSatisfy({if case .object(let row)=$0 {return row["source"]?.stringValue.map(available.contains) ?? false};return false}) else {throw DataFlowContractError("Choose source columns from the connected table.")}
            }
            let filename=string(config["databaseFilename"],fallback:"analytics.sqlite")
            guard !filename.isEmpty, filename != ".", filename != "..", filename.utf8.count<=200, !filename.contains(where:{"/\\\0".contains($0)}) else {throw DataFlowContractError("Choose a SQLite database filename without a path.")}
            return config
        }
        let format=string(config["format"],fallback:"csv")
        guard formats.contains(where:{$0.0==format}) else {throw DataFlowContractError("Choose a supported Table Output format.")}
        var options=defaults(format)
        if let value=config["options"] {
            guard case .object(let given)=value,Set(given.keys).isSubset(of:Set(options.keys)) else {throw DataFlowContractError("Options must belong to the selected format.")}
            options.merge(given,uniquingKeysWith:{$1})
        }
        for field in fields(format) {
            let value=options[field.id]
            switch field.kind {
            case .text:guard case .string = value else {throw DataFlowContractError("\(field.title) requires text.")}
            case .boolean:guard case .bool = value else {throw DataFlowContractError("\(field.title) requires a checkbox value.")}
            case .integer:guard case .integer(let number)=value,number>=1,number<=16_777_216 else {throw DataFlowContractError("\(field.title) must be between 1 and 16,777,216.")}
            case .optionalInteger:guard value == .null || {if case .integer = value {return true};return false}() else {throw DataFlowContractError("Enter a whole compression level or leave it blank.")}
            case .choices(let choices):guard choices.contains(where:{$0.0==string(value)}) else {throw DataFlowContractError("Choose a supported \(field.title.lowercased()).")}
            }
        }
        if ["csv","tsv"].contains(format) {
            let separator=string(config["separator"])
            func character(_ text:String)->Bool {text.unicodeScalars.count==1 && !text.contains(where:{$0=="\n" || $0=="\r" || $0=="\0"})}
            guard character(separator),character(string(options["quoteChar"])),character(string(options["escapeChar"])) else {throw DataFlowContractError("Delimiter, quote and escape must each be one non-newline character.")}
            guard separator != string(options["quoteChar"]),options["escaping"] != .string("escape") || ![separator,string(options["quoteChar"])].contains(string(options["escapeChar"])) else {throw DataFlowContractError("Delimiter, quote and escape characters must differ.")}
            guard options["quoting"] != .string("none") || options["escaping"] == .string("escape") else {throw DataFlowContractError("No quoting requires Escape character handling.")}
            if options["numericFormat"] == .string("custom") {guard string(options["numericPattern"]).range(of:#"^\.?[0-9]{0,2}[eEfFgG]$"#,options:.regularExpression) != nil else {throw DataFlowContractError("Use a numeric pattern such as .6g or .2f.")}}
        }
        if format == "json",options["shape"] == .string("rowsSchema"),options["nullPolicy"] == .string("omit") {throw DataFlowContractError("Rows plus schema requires positional JSON nulls.")}
        if format == "parquet",case .integer(let level)=options["compressionLevel"] {
            let bounds:[String:ClosedRange<Int64>]=["gzip":0...9,"brotli":0...11,"zstd":(-7)...22]
            guard let range=bounds[string(options["compression"])],range.contains(level) else {throw DataFlowContractError("That compression level is unsupported for the selected codec.")}
        }
        if format == "xlsx" {
            let sheet=string(options["sheet"]),cell=string(options["startCell"])
            guard !sheet.isEmpty,sheet.count<=31,!sheet.contains(where:{"\\/*?:[]".contains($0)}),!sheet.hasPrefix("'"),!sheet.hasSuffix("'") else {throw DataFlowContractError("Use a valid Excel sheet name of at most 31 characters.")}
            guard cell.range(of:#"^[A-Z]{1,3}[1-9][0-9]{0,6}$"#,options:.regularExpression) != nil else {throw DataFlowContractError("Use an Excel start cell such as A1.")}
        }
        if config["imagePolicy"] == .string("copyAssets"),config["overwrite"] == .bool(true) {throw DataFlowContractError("Copied-assets export requires a new bundle name with Stop if existing.")}
        return config
    }
    static func string(_ value:JSONValue?,fallback:String="")->String {if case .string(let text)=value {return text};return fallback}
}
