import h5py
import numpy as np
import json
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import random
from numpy.linalg import norm
import datetime

MB_DIST = {
    "iar <*> dear <*>": 1,
    "<*> total interrupts. <*> critical input interrupts. <*> microseconds total spent on critical input interrupts, <*> microseconds max time in a critical input interrupt.": 2,
    "idoproxydb hit ASSERT condition: ASSERT expression=<*> Source file=<*> Source line=<*> Function=<*> IdoTransportMgr::SendPacket(IdoUdpMgr<*>, BglCtlPavTrace<*>)": 3,
    "<*> microseconds spent in the rbs signal handler during <*> calls. <*> microseconds was the maximum time for a single instance of a correctable ddr.": 4,
    "generating <*>": 5,
    "fpr<*>=<*> <*> <*> <*>": 6,
    "ciod: LOGIN chdir(<*>) failed: No such file or directory": 7,
    "ciod: generated <*> core files for program <*>": 8,
    "<*> TLB error interrupt": 9,
    "data storage interrupt": 10,
    "CE sym <*> at <*> mask <*>": 11,
    "rts: kernel terminated for reason <*>": 12,
    "data cache <*> parity error detected. attempting to correct": 13,
    "<*> ddr error(s) detected and corrected on rank <*> symbol <*> over <*> seconds": 14,
    "<*> floating point alignment exceptions": 15,
    "floating point <*>": 16,
    "<*> double-hummer alignment exceptions": 17,
    "byte ordering exception.....................0": 18,
    "Node card VPD check: <*> node in processor card slot <*> do not match. VPD ecid <*> found <*>": 19,
    "ciod: Error loading <*> invalid or missing program image, Exec format error": 20,
    "instruction cache parity error corrected": 21,
    "Can not get assembly information for node card": 22,
    "ciod: Error reading message prefix on CioStream socket to <*> Link has been severed": 23,
    "ciod: Error loading <*> invalid or missing program image, No such file or directory": 24,
    "Error receiving packet on tree network, expecting type <*> instead of type <*> <*> <*> <*> <*> <*> <*> <*> <*>": 25,
    "Node card status: no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JTAG Reset is asserted. ASIC JTAG Reset is asserted. Temperature Mask is not active. No temperature error. Temperature Limit Error Latch is clear. PGOOD IS NOT ASSERTED. PGOOD ERROR LATCH IS ACTIVE. MPGOOD IS NOT OK. MPGOOD ERROR LATCH IS ACTIVE. The 2.5 volt rail is OK. The 1.5 volt rail is OK.": 26,
    "Microloader Assertion": 27,
    "instruction address: <*>": 28,
    "<*> tree receiver <*> in re-synch state event(s) (dcr <*> detected": 29,
    "Kernel detected <*> integer alignment exceptions <*> iar <*> dear <*> <*> iar <*> dear <*> <*> iar <*> dear <*> <*> iar <*> dear <*> <*> iar <*> dear <*> <*> iar <*> dear <*> <*> iar <*> dear <*> <*> iar <*> dear <*>": 30,
    "data store interrupt caused by <*>": 31,
    "<*> ddr errors(s) detected and corrected on rank <*> symbol <*>, bit <*>": 32,
    "Lustre mount FAILED : <*> : point /p/gb1": 33,
    "<*> torus receiver <*> input pipe error(s) (dcr <*> detected and corrected": 34,
    "exception <*>": 35,
    "ciod: Received signal <*> <*> <*> <*>": 36,
    "suppressing further interrupts of same type": 37,
    "total of <*> ddr error(s) detected and corrected": 38,
    "machine check: i-fetch......................0": 39,
    "store <*>": 40,
    "program interrupt: trap <*>": 41,
    "program interrupt: illegal <*>": 42,
    "ciod: Error reading message prefix after <*> on CioStream socket to <*> <*> <*> <*> <*>": 43,
    "ciod: pollControlDescriptors: Detected the debugger died.": 44,
    "ciod: Error loading <*> invalid or missing program image, Permission denied": 45,
    "wait state enable.................0": 46,
    "program interrupt: fp cr <*>": 47,
    "<*> tree receiver <*> in re-synch state event(s) (dcr <*> detected over <*> seconds": 48,
    "<*>:<*> <*>:<*> <*>:<*> <*>:<*>": 49,
    "New ido chip inserted into the database: <*> <*> <*> <*>": 50,
    "debug interrupt enable............0": 51,
    "program interrupt: unimplemented operation..0": 52,
    "ciod: LOGIN <*> failed: Permission denied": 53,
    "disable store gathering..................0": 54,
    "auxiliary processor.........................0": 55,
    "rts tree/torus link training failed: wanted: <*> got: <*>": 56,
    "critical input interrupt <*>": 57,
    "program interrupt: fp compare...............0": 58,
    "Machine State Register: <*>": 59,
    "machine check <*>": 60,
    "external input interrupt <*>": 61,
    "guaranteed <*> cache block <*>": 62,
    "problem state <*>": 63,
    "debug wait enable.................0": 64,
    "program interrupt: privileged instruction...0": 65,
    "NFS Mount failed on <*> slept <*> seconds, retrying <*>": 66,
    "data address space................0": 67,
    "Ido chip status changed: <*> <*> <*> <*> <*> <*> <*> <*> <*> <*> <*>": 68,
    "Node card is not fully functional": 69,
    "rts: bad message header: expecting type <*> instead of type <*> <*> <*> <*> <*> <*> <*> <*> <*>": 70,
    "ciod: Error creating node map from file <*> Permission denied": 71,
    "rts: bad message header: cpu <*> invalid <*> <*> <*> <*> <*> <*> <*> <*>": 72,
    "core configuration register: <*>": 73,
    "PrepareForService shutting down <*> as part of Service Action <*>": 74,
    "total of <*> ddr error(s) detected and corrected over <*> seconds": 75,
    "disable write lines 2:4..................0": 76,
    "data address: <*>": 77,
    "ciod: failed to read message prefix on control stream (CioStream socket to <*>": 78,
    "General Purpose Registers:": 79,
    "<*> L3 <*> error(s) (dcr <*> detected and corrected": 80,
    "ciod: Error creating node map from file <*> Block device required": 81,
    "L1 DCACHE summary averages: #ofDirtyLines: <*> out of 1024 #ofDirtyDblWord: <*> out of 4096": 82,
    "Ido packet timeout": 83,
    "instruction address space.........0": 84,
    "program interrupt: imprecise exception......0": 85,
    "force load/store alignment...............0": 86,
    "Found invalid node ecid in processor card slot <*> ecid 0000000000000000000000000000": 87,
    "icache prefetch <*>": 88,
    "ciod: Error opening node map file <*> No such file or directory": 89,
    "<*> torus receiver <*> input pipe error(s) (dcr <*> detected and corrected over <*> seconds": 90,
    "rts internal error": 91,
    "capture <*>": 92,
    "disable trace broadcast..................0": 93,
    "program interrupt": 94,
    "d-cache <*> parity <*>": 95,
    "ciod: Error creating node map from file <*> Bad address": 96,
    "<*> torus <*> <*> <*> error(s) (dcr <*> detected and corrected over <*> seconds": 97,
    "ciod: Missing or invalid fields on line <*> of node map file <*>": 98,
    "CHECK_INITIAL_GLOBAL_INTERRUPT_VALUES": 99,
    "rts panic! - stopping execution": 100,
    "Temperature Over Limit on link card": 101,
    "ciod: Error creating node map from file <*> Resource temporarily unavailable": 102,
    "EndServiceAction <*> performed upon <*> by <*>": 103,
    "<*> PGOOD error latched on link card": 104,
    "ciod: Error reading message prefix on CioStream socket to <*> Connection reset by peer": 105,
    "MidplaneSwitchController performing bit sparing on <*> bit <*>": 106,
    "disable apu instruction broadcast........0": 107,
    "Lustre mount FAILED : <*> : block_id : location": 108,
    "ciod: cpu <*> at treeaddr <*> sent unrecognized message <*>": 109,
    "<*> L3 <*> error(s) (dcr <*> detected and corrected over <*> seconds": 110,
    "ciod: Error creating node map from file <*> Bad file descriptor": 111,
    "enabled exception summary................0": 112,
    "memory manager <*>": 113,
    "ciod: Error loading <*> not a CNK program image": 114,
    "parity error in bank <*>": 115,
    "number of correctable errors detected in L3 <*>": 116,
    "rts: bad message header: packet index <*> greater than max 366 <*> <*> <*> <*> <*> <*> <*> <*>": 117,
    "interrupt threshold...0": 118,
    "<*> torus <*> <*> <*> error(s) (dcr <*> detected and corrected": 119,
    "ciod: Error reading message prefix on CioStream socket to <*> Connection timed out": 120,
    "plus <*>": 121,
    "Node card VPD check: missing <*> node, VPD ecid <*> in processor card slot <*>": 122,
    "ciod: <*> coordinate <*> exceeds physical dimension <*> at line <*> of node map file <*>": 123,
    "ciod: LOGIN chdir(<*>) failed: Input/output error": 124,
    "MidplaneSwitchController::receiveTrain() iap failed: <*> <*> <*>": 125,
    "ddr: activating redundant bit steering for next allocation: <*> <*>": 126,
    "r<*>=<*> r<*>=<*> r<*>=<*> r<*>=<*>": 127,
    "Special Purpose Registers:": 128,
    "Power deactivated: <*>": 129,
    "round nearest............................0": 130,
    "memory and bus summary...........................0": 131,
    "<*> power module <*> is not accessible": 132,
    "Error: unable to mount filesystem": 133,
    "ciod: Error creating node map from file <*> No child processes": 134,
    "data <*> plb <*>": 135,
    "kernel panic": 136,
    "DDR failing info register: <*>": 137,
    "Missing reverse cable: Cable <*> <*> <*> <*> --> <*> <*> <*> <*> is present BUT the reverse cable <*> <*> <*> <*> --> <*> <*> <*> <*> is missing": 138,
    "msr=<*> dear=<*> esr=<*> fpscr=<*>": 139,
    "ciod: for node <*> read continuation request but ioState is <*>": 140,
    "round toward <*>": 141,
    "ciod: Error loading <*> program image too big, <*> > <*>": 142,
    "EndServiceAction is restarting the <*> in midplane <*> as part of Service Action <*>": 143,
    "DDR failing data registers: <*> <*>": 144,
    "DDR failing <*> register: <*> <*>": 145,
    "ciodb has been restarted.": 146,
    "ddr: Unable to steer <*> <*> - rank is already steering symbol <*> Due to multiple symbols being over the correctable": 147,
    "ciod: Message code <*> is not <*> or 4294967295": 148,
    "invalid <*>": 149,
    "ciod: Error creating node map from file <*> Device or resource busy": 150,
    "minus <*>": 151,
    "dbcr0=<*> dbsr=<*> ccr0=<*>": 152,
    "ddr: activating redundant bit steering: <*> <*>": 153,
    "uncorrectable <*>": 154,
    "number of lines with parity errors written to L3 <*>": 155,
    "rts: bad message header: index 0 greater than total 0 <*> <*> <*> <*> <*> <*> <*> <*>": 156,
    "max number of outstanding prefetches.....7": 157,
    "ddr: Unable to steer <*> <*> - rank is already steering symbol <*> Due to multiple symbols being over the correctable error threshold, consider replacing the card": 158,
    "fraction <*>": 159,
    "ciod: duplicate canonical-rank <*> to logical-rank <*> mapping at line <*> of node map file <*>": 160,
    "ddr: Suppressing further CE interrupts": 161,
    "ciod: In packet from node <*> <*> message code <*> is not <*> or 4294967295 <*> <*> <*> <*>": 162,
    "mmcs_db_server has been started: ./mmcs_db_server --useDatabase BGL --dbproperties db.properties --iolog /bgl/BlueLight/logs/BGL --reconnect-blocks all --shutdown-timeout 120 --shutdown-timeout 240": 163,
    "disable speculative access...............0": 164,
    "MidplaneSwitchController::sendTrain() port disconnected: <*>": 165,
    "instruction plb <*>": 166,
    "quiet NaN................................0": 167,
    "PrepareForService is being done on this Midplane <*> <*> <*> by <*>": 168,
    "Node card status: ALERT 0, ALERT 1, ALERT 2, ALERT 3 is (are) active. Clock Mode is Low. Clock Select is Midplane. Phy JTAG Reset is asserted. ASIC JTAG Reset is not asserted. Temperature Mask is not active. No temperature error. Temperature Limit Error Latch is clear. PGOOD is asserted. PGOOD error latch is clear. MPGOOD is OK. MPGOOD error latch is clear. The 2.5 volt rail is OK. ": 169,
    "ciod: Unexpected eof at line <*> of node map file <*>": 170,
    "EndServiceAction is restarting the <*> cards in Midplane <*> as part of Service Action <*>": 171,
    "prefetch depth for PLB slave.............1": 172,
    "correctable <*>": 173,
    "shutdown complete": 174,
    "DCR <*> : <*>": 175,
    "job <*> timed out. Block freed.": 176,
    "Node card status: no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JTAG Reset is asserted. ASIC JTAG Reset is not asserted. Temperature Mask is not active. No temperature error. Temperature Limit Error Latch is clear. PGOOD IS NOT ASSERTED. PGOOD ERROR LATCH IS ACTIVE. MPGOOD IS NOT OK. MPGOOD ERROR LATCH IS ACTIVE. The 2.5 volt rail is OK. The 1.5 volt rail is": 177,
    "regctl scancom interface.........................0": 178,
    "imprecise machine <*>": 179,
    "inexact <*>": 180,
    "underflow <*>": 181,
    "minus <*> <*>": 182,
    "summary...........................1": 183,
    "qw trapped............0": 184,
    "divide-by-zero <*>": 185,
    "prefetch depth for core <*>": 186,
    "mmcs_db_server has been started: ./mmcs_db_server --useDatabase BGL --dbproperties serverdb.properties --iolog /bgl/BlueLight/logs/BGL --reconnect-blocks all --shutdown-timeout <*>": 187,
    "tlb <*>": 188,
    "Bad cable going into LinkCard <*> Jtag <*> Port <*> - <*> bad wires": 189,
    "i-cache parity error..............0": 190,
    "start <*>": 191,
    "write buffer commit threshold............2": 192,
    "ciod: for node <*> incomplete data written to core file <*>": 193,
    "LinkCard is not fully functional": 194,
    "size of DDR we are caching...............1 (512M)": 195,
    "<*> <*> <*> BGLERR_IDO_PKT_TIMEOUT connection lost to node/link/service card": 196,
    "L3 <*> <*> register: <*>": 197,
    "Hardware monitor caught java.net.SocketException: Broken pipe and is stopping": 198,
    "mmcs_db_server has been started: ./mmcs_db_server --useDatabase BGL --dbproperties serverdb.properties --iolog /bgl/BlueLight/logs/BGL --reconnect-blocks all": 199,
    "enable <*> exceptions.........0": 200,
    "Node card status: no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JTAG Reset is asserted. ASIC JTAG Reset is asserted. Temperature Mask is not active. No temperature error. Temperature Limit Error Latch is clear. PGOOD IS NOT ASSERTED. PGOOD ERROR LATCH IS ACTIVE. MPGOOD IS NOT OK. MPGOOD ERROR LATCH IS ACTIVE. THE 2.5 VOLT RAIL IS NOT OK. THE 1.5 VOLT RAIL IS": 201,
    "mask..................<*>": 202,
    "command manager unit summary.....................0": 203,
    "<*> torus non-crc error(s) (dcr <*> detected and corrected": 204,
    "NodeCard is not fully functional": 205,
    "PrepareForService is being done on this <*> <*> <*> <*> <*> by <*>": 206,
    "reserved.................................0": 207,
    "overflow exception.......................0": 208,
    "miscompare............0": 209,
    "idoproxydb hit ASSERT condition: ASSERT expression=pTargetMgr Source file=idoclientmgr.cpp Source line=353 Function=int IdoClientMgr::TargetClose(const char<*>)": 210,
    "idoproxy communication failure: socket closed": 211,
    "MailboxMonitor::serviceMailboxes() lib_ido_error: -1019 socket closed": 212,
    "While setting fan speed caught java.lang.IllegalStateException: while executing I2C Operation caught <*>": 213,
    "parity error in read queue <*>": 214,
    "ciod: Error creating node map from file <*> Cannot allocate memory": 215,
    "<*> <*> Status(A)": 216,
    "Power Good signal deactivated: <*> A service action may be required.": 217,
    "Running as background command": 218,
    "this link card is not fully functional": 219,
    "idoproxydb has been started: $Name: <*> $ Input parameters: -enableflush -loguserinfo db.properties BlueGene1": 220,
    "disable flagging of DDR UE's as major internal error.0": 221,
    "ddr: Unable to steer <*> <*> - rank is already steering symbol <*> Due to multiple symbols being over the correctable e": 222,
    "chip select...........0": 223,
    "parity error in write buffer...................0": 224,
    "monitor caught java.lang.IllegalStateException: while executing <*> Operation caught java.net.SocketException: Broken pipe and is stopping": 225,
    "Target=<*> Message=<*>": 226,
    "Starting SystemController": 227,
    "close EDRAM pages as soon as possible....0": 228,
    "symbol................<*>": 229,
    "error threshold, consider replacing the card": 230,
    "ciod: Error creating node map from file <*> No such file or directory": 231,
    "BGLMaster has been started: ./BGLMaster --consoleip 127.0.0.1 --consoleport 32035 --configfile bglmaster.init": 232,
    "state machine....................................0": 233,
    "ciod: In packet from node <*> <*> message still ready for node <*> <*> <*> <*> <*>": 234,
    "ciod: Error creating node map from file <*> Argument list too long": 235,
    "Controlling BG/L rows [ <*> ]": 236,
    "disable all access to cache directory....0": 237,
    "DDR machine check register: <*> <*>": 238,
    "Core Configuration Register 0: <*>": 239,
    "address parity error..0": 240,
    "mmcs_server exited abnormally due to signal: Segmentation fault": 241,
    "Hardware monitor caught java.lang.IllegalStateException: while executing I2C Operation caught java.net.SocketException: Broken pipe and is stopping": 242,
    "VALIDATE_LOAD_IMAGE_CRC_IN_DRAM": 243,
    "While initializing node card, ido with LP of <*> caught java.lang.IllegalStateException: IDo is not in functional state -- currently in state COMMUNICATION_ERROR": 244,
    "enable invalid operation exceptions......0": 245,
    "Node card status: no ALERTs are active. Clock Mode is Low. Clock Select is Midplane. Phy JTAG Reset is asserted. ASIC JTAG Reset is asserted. Temperature Mask is not active. No temperature error. Temperature Limit Error Latch is clear. PGOOD IS NOT ASSERTED. PGOOD ERROR LATCH IS ACTIVE. MPGOOD is OK. MPGOOD ERROR LATCH IS ACTIVE. The 2.5 volt rail is OK. The 1.5 volt rail is OK.": 246,
    "enable non-IEEE mode.....................0": 247,
    "ddr: excessive soft failures, consider replacing the card": 248,
    "<*> exited normally with exit code <*>": 249,
    "size of scratchpad portion of L3.........0 (0M)": 250,
    "PrepareForService is being done on this rack <*> by <*>": 251,
    "idoproxydb hit ASSERT condition: ASSERT expression=!(nMsgLen > 0x10000) Source file=idomarshalerio.cpp Source line=1929 Function=int IdoMarshalerRecvBuffer::ReadBlock(IdoMsg::IdoMsgHdr<*>&)": 252,
    "<*> exited abnormally due to signal: Aborted": 253,
    "Expected 10 active FanModules, but found 9 ( Found <*> <*> <*> <*> <*> <*> <*> <*> <*> ).": 254,
    "MidplaneSwitchController::clearPort() bll_clear_port failed: <*>": 255,
    "BGLMaster has been started: ./BGLMaster --consoleip 127.0.0.1 --consoleport 32035 --configfile bglmaster.init --autorestart y": 256,
    "NodeCard VPD is corrupt": 257,
    "MidplaneSwitchController::parityAlignment() pap failed: <*> <*> <*>": 258,
    "Error getting detailed hw info for node, caught java.io.IOException: Problems with the chip, could not pull all resets": 259,
    "ciod: Error loading <*> invalid or missing program image, No such device": 260,
    "<*> correctable errors exceeds threshold (iar <*> lr <*>": 261,
    "mmcs_db_server has been started: ./mmcs_db_server --useDatabase BGL --dbproperties db.properties --iolog /bgl/BlueLight/logs/BGL --reconnect-blocks all --shutdown-timeout 120": 262,
    "NodeCard temperature sensor chip <*> is not accessible": 263,
    "no ethernet link": 264,
    "turn on hidden refreshes.................1": 265,
    "ddr: redundant bit steering failed, sequencer timeout": 266,
    "While reading FanModule caught java.lang.IllegalStateException: while executing I2C Operation caught <*>": 267,
    "monitor caught java.lang.IllegalStateException: while executing CONTROL Operation caught java.io.EOFException and is stopping": 268,
    "Torus non-recoverable error DCRs follow.": 269,
    "L3 major internal error": 270,
    "mmcs_db_server has been started: ./mmcs_db_server --useDatabase BGL --dbproperties serverdb.properties --iolog /bgl/BlueLight/logs/BGL --reconnect-blocks all --no-reconnect-blocks": 271,
    "No power module <*> found found on link card": 272,
    "Added <*> subnets and <*> addresses to DB": 273,
    "Error sending packet on tree network, packet at address <*> is not aligned": 274,
    "ciod: sendMsgToDebugger: error sending PROGRAM_EXITED message to debugger.": 275,
    "Problem communicating with service card, ido chip: <*> java.lang.IllegalStateException: IDo is not in functional state -- currently in state COMMUNICATION_ERROR": 276,
    "BglIdoChip table has <*> IDOs with the same IP address <*>": 277,
    "power module status fault detected on node card. status registers are: <*>": 278,
    "Error getting detailed hw info for node, caught java.io.IOException: Problems with the chip, could not enable clock domains": 279,
    "rts assertion failed: `personality->version == BGLPERSONALITY_VERSION' in `void start()' at start.cc:131": 280,
    "machine state register:": 281,
    "Error receiving packet on tree network, packet index <*> greater than max 366 <*> <*> <*> <*> <*> <*> <*> <*>": 282,
    "MailboxMonitor::serviceMailboxes() lib_ido_error: -1114 unexpected socket error: Broken pipe": 283,
    "NodeCard VPD chip is not accessible": 284,
    "While inserting monitor info into DB caught java.lang.NullPointerException": 285,
    "Error getting detailed hw info for node, caught java.io.IOException: Problems with the chip, clear all resets": 286,
    "monitor caught java.lang.UnsupportedOperationException: power module <*> not present and is stopping": 287,
    "mmcs_db_server has been started: /bgl/BlueLight/ppcfloor/bglsys/bin/mmcs_db_server --useDatabase BGL --dbproperties serverdb.properties --iolog /bgl/BlueLight/logs/BGL --reconnect-blocks all": 288,
    "Node card status: ALERT 0, ALERT 1, ALERT 2, ALERT 3 is (are) active. Clock Mode is Low. Clock Select is Midplane. Phy JTAG Reset is asserted. ASIC JTAG Reset is not asserted. TEMPERATURE MASK IS ACTIVE. No temperature error. Temperature Limit Error Latch is clear. PGOOD is asserted. PGOOD error latch is clear. MPGOOD is OK. MPGOOD error latch is clear. The 2.5 volt rail is OK. The ": 289,
    "parity error.......0": 290,
    "<*> torus non-crc error(s) (dcr <*> detected and corrected over <*> seconds": 291,
    "DeclareServiceNetworkCharacteristics has been run with the force option but the DB is not empty": 292,
    "ddrSize == <*> || ddrSize == <*>": 293
}
BATSH_SIZE = 10000
client = OpenAI(
    api_key="sk-key",
    base_url="https://api.deepseek.com"
)

def deepseek_rag_inference(llm_content, topk_logs, model_preds):
    """
    输入:
        llm_content: 当前日志内容 (str)
        topk_logs: list[{content, label}]
        model_preds: dict{model_name: {pred, conf}}

    返回:
        {"pred_label": int, "confidence": float}
    """

    rag_input = {
        "llm_content": llm_content,
        "similar_logs": topk_logs,
        "model_predictions": model_preds
    }
    # print(rag_input)
    # 3. 严格按 JSON 格式返回如下内容，不要输出其他内
    prompt = f"""
    你是一个日志异常检测模型,准备对当前日志进行是否异常的检测,同时你还将获得通过RAG获取到的内容。
    当前待检测的日志为llm_content,以下输入为 RAG 获取的内容:
    相似日志信息和其对应的标签在similar_logs中以字典数组的形式表示:
    {{{{'content': 相似日志1, 'label': 相似日志1的标签, 'similarity': 相似日志1的相似度}},{{'content': 相似日志2, 'label': 相似日志2的标签, 'similarity': 相似日志2的相似度}},......}})，
    以及 6 个传统机器学习模型的预测结果，包括每个模型的预测结果和每个模型的置信度,存储在model_predictions中:
    {{机器学习模型1: {{'pred': 模型1预测结果, 'conf': 模型1预测结果置信度}}, 机器学习模型2: {{'pred': 模型2预测结果, 'conf': 模型2预测结果置信度}}, ......}}
    请使用集成的方式统一分析六个模型的预测结果
    你的任务：
    1. 基于输入内容推断该日志是否异常（0 = 正常，1 = 异常）
    2. 以 0~1 的浮点数给出你的置信度
    3. 严格按 JSON 格式返回如下内容，并输出判断理由：

    {{
    "pred_label": 0 或 1,
    "confidence": 0.0 ~ 1.0
    "reasoning": "你的判断理由"
    }}

    以下是输入 JSON：

    {json.dumps(rag_input, indent=2, ensure_ascii=False)}
    """

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是专业的日志异常检测模型，只输出结构化 JSON。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # 解析 JSON 输出
    text = resp.choices[0].message.content
    # print(text)

    try:
        result = json.loads(text)
        return result
    except Exception:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        clean_text = text[json_start:json_end]
        return json.loads(clean_text)

def get_mb_folder(template_text, MB_DIST):
    """
    根据 template_text 进行前缀匹配（或包含匹配），返回 MB 编号。
    如果没有匹配到，返回默认值 0。
    """
    tmpl = str(template_text).strip()

    for key in MB_DIST:
        if tmpl.startswith(key):
            return MB_DIST[key]

    for key in MB_DIST:
        if key in tmpl:
            return MB_DIST[key]

    return 0

def map_index_to_cluster_id(template_dir, nearest_idx):
    report_file = os.path.join(
        template_dir,
        f"dbscan_results/{os.path.basename(template_dir)}_cluster_report.json"
    )

    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    # JSON key 就是真实 cluster id（字符串），需要按数字排序
    real_cluster_ids = sorted(report.keys(), key=lambda x: int(x))

    if nearest_idx >= len(real_cluster_ids):
        raise ValueError(
            f"nearest_idx {nearest_idx} 超过簇数量 {len(real_cluster_ids)}"
        )

    # 返回真实簇 ID （int）
    return int(real_cluster_ids[nearest_idx])

def find_nearest_cluster(query_embedding, template_dir):
    centers_file = os.path.join(template_dir, f"dbscan_results/{os.path.basename(template_dir)}_cluster_centers.npy")
    if not os.path.exists(centers_file):
        raise FileNotFoundError(f"未找到聚类中心文件: {centers_file}")
    centers = np.load(centers_file)
    dists = np.linalg.norm(centers - query_embedding, axis=1)
    nearest_idx = np.argmin(dists)
    cluster_id = map_index_to_cluster_id(template_dir, nearest_idx)
    return cluster_id, centers[nearest_idx]

# def load_cluster_topk(template_dir, cluster_id):
#     vec_file = os.path.join(template_dir, f"dbscan_results/cluster_{cluster_id}_topk_vectors.npy")
#     logs_file = os.path.join(template_dir, f"dbscan_results/{os.path.basename(template_dir)}_cluster_top_logs.json")

#     if not os.path.exists(vec_file) or not os.path.exists(logs_file):
#         return np.array([]), []

#     topk_vectors = np.load(vec_file)
#     print(logs_file)
#     with open(logs_file, "r", encoding="utf-8") as f:
#         all_logs = json.load(f)

#     cluster_logs = all_logs.get(str(cluster_id), [])
#     return topk_vectors, cluster_logs

def load_multi_cluster_logs(template_dir, cluster_ids):
    logs_file = os.path.join(
        template_dir,
        f"dbscan_results/{os.path.basename(template_dir)}_cluster_top_logs.json"
    )

    if not os.path.exists(logs_file):
        return []

    with open(logs_file, "r", encoding="utf-8") as f:
        all_logs = json.load(f)

    collected = []

    for cid in cluster_ids:
        logs = all_logs.get(str(cid), [])
        label = get_cluster_label(template_dir, cid)
        for log in logs:
            collected.append({
                "content": log,
                "label": label
            })

    return collected

def get_cluster_label(template_dir, cluster_id):
    report_file = os.path.join(
        template_dir,
        f"dbscan_results/{os.path.basename(template_dir)}_cluster_report.json"
    )

    if not os.path.exists(report_file):
        raise FileNotFoundError(f"未找到文件: {report_file}")

    with open(report_file, "r", encoding="utf-8") as f:
        cluster_report = json.load(f)

    cluster_info = cluster_report.get(str(cluster_id), None)
    if cluster_info is None:
        raise ValueError(f"簇 {cluster_id} 在报告中不存在")

    label = next(iter(cluster_info.keys()))
    return int(label)

def update_topk(center, query_embedding, query_content, topk_vectors, cluster_logs, top_k=10):
    """
    更新 top-k 最近日志，并返回更新前的 top-k
    """
    # 保存更新前的 top-k
    previous_topk = [{"content": c, "label": None} for c in cluster_logs]  # label 暂时先为 None

    # 计算距离
    dists = np.linalg.norm(topk_vectors - center, axis=1) if len(topk_vectors) > 0 else np.array([])

    if len(topk_vectors) < top_k:
        topk_vectors = np.vstack([topk_vectors, query_embedding]) if len(topk_vectors) > 0 else np.array([query_embedding])
        cluster_logs.append(query_content)
    else:
        farthest_idx = np.argmax(dists)
        farthest_dist = dists[farthest_idx]
        query_dist = np.linalg.norm(query_embedding - center)
        if query_dist < farthest_dist:
            topk_vectors[farthest_idx] = query_embedding
            cluster_logs[farthest_idx] = query_content

    return topk_vectors, cluster_logs, previous_topk

def save_updated_topk(template_dir, cluster_id, topk_vectors, cluster_logs):
    vec_file = os.path.join(template_dir, f"dbscan_results/cluster_{cluster_id}_topk_vectors.npy")
    logs_file = os.path.join(template_dir, f"dbscan_results/{os.path.basename(template_dir)}_cluster_top_logs.json")

    # 保存 topk_vectors 为 npy 文件
    np.save(vec_file, topk_vectors)

    # 将 cluster_logs 中的 ndarray 转为列表
    cluster_logs_serializable = []
    for log in cluster_logs:
        if isinstance(log, np.ndarray):
            cluster_logs_serializable.append(log.tolist())
        else:
            cluster_logs_serializable.append(log)

    # 更新 JSON 文件
    if os.path.exists(logs_file):
        with open(logs_file, "r", encoding="utf-8") as f:
            all_logs = json.load(f)
    else:
        all_logs = {}

    all_logs[str(cluster_id)] = cluster_logs_serializable
    with open(logs_file, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, indent=2, ensure_ascii=False)

def pick_random_clusters(template_dir, num_clusters=10):
    report_file = os.path.join(
        template_dir,
        f"dbscan_results/{os.path.basename(template_dir)}_cluster_report.json"
    )

    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    all_cluster_ids = sorted(report.keys(), key=lambda x: int(x))

    if len(all_cluster_ids) == 0:
        raise ValueError("没有簇可供选择！")

    # 不足 10 个就全选
    picked = random.sample(all_cluster_ids, min(num_clusters, len(all_cluster_ids)))

    # 转成 int
    return [int(x) for x in picked]

# def process_query(query_embedding, query_content, template_dir, top_k=10):
#     """
#     处理单条 query embedding
#     返回:
#         cluster_id: 最近簇 ID
#         cluster_label: 簇对应 label
#         previous_topk: 更新前的 top-k 日志列表，每条日志是 {"content": ..., "label": ...}
#     """
#     cluster_id, nearest_center = find_nearest_cluster(query_embedding, template_dir)
#     topk_vectors, cluster_logs = load_cluster_topk(template_dir, cluster_id)
    
#     cluster_label = get_cluster_label(template_dir, cluster_id)

#     topk_vectors, cluster_logs, previous_topk = update_topk(
#         nearest_center, query_embedding, query_content, topk_vectors, cluster_logs, top_k
#     )

#     # 更新 JSON 文件
#     save_updated_topk(template_dir, cluster_id, topk_vectors, cluster_logs)

#     # 给 previous_topk 补上簇 label
#     for log in previous_topk:
#         log["label"] = cluster_label

#     return cluster_id, cluster_label, previous_topk

# def process_query(query_embedding, query_content, template_dir, top_k=10):
#     """
#     新逻辑：
#     1. 随机选择同模板目录下的 10 个簇中心
#     2. 返回这些簇中所有日志（带簇标签）
#     3. 不更新 top-k，不写回 JSON
#     """

#     # -----------------------------
#     # 1. 读取簇报告，随机取 10 个簇
#     # -----------------------------
#     report_file = os.path.join(
#         template_dir,
#         f"dbscan_results/{os.path.basename(template_dir)}_cluster_report.json"
#     )

#     if not os.path.exists(report_file):
#         raise FileNotFoundError(f"未找到簇报告文件: {report_file}")

#     with open(report_file, "r", encoding="utf-8") as f:
#         cluster_report = json.load(f)

#     all_cluster_ids = sorted(cluster_report.keys(), key=lambda x: int(x))

#     if len(all_cluster_ids) == 0:
#         raise ValueError("当前模板目录没有任何簇，无法检索日志")

#     # 随机选择 10 个簇（不足 10 个就全部）
#     picked_cluster_ids = random.sample(
#         all_cluster_ids,
#         min(10, len(all_cluster_ids))
#     )

#     # 转换为 int
#     picked_cluster_ids = [int(cid) for cid in picked_cluster_ids]

#     # -----------------------------
#     # 2. 读取日志文件，收集所有选择簇的日志
#     # -----------------------------
#     logs_file = os.path.join(
#         template_dir,
#         f"dbscan_results/{os.path.basename(template_dir)}_cluster_top_logs.json"
#     )

#     if not os.path.exists(logs_file):
#         raise FileNotFoundError(f"未找到簇日志文件: {logs_file}")

#     with open(logs_file, "r", encoding="utf-8") as f:
#         all_logs = json.load(f)

#     previous_topk = []

#     # 遍历所有随机簇，收集日志
#     for cid in picked_cluster_ids:
#         str_cid = str(cid)

#         # 获取簇标签
#         cluster_info = cluster_report.get(str_cid, None)
#         if cluster_info is None:
#             continue

#         cluster_label = int(next(iter(cluster_info.keys())))

#         # 获取簇日志
#         cluster_logs = all_logs.get(str_cid, [])

#         for log in cluster_logs:
#             previous_topk.append({
#                 "content": log,
#                 "label": cluster_label
#             })

#     # -----------------------------
#     # 3. 为兼容旧逻辑，返回随机簇中的第一个簇标签
#     # -----------------------------
#     first_cluster_id = picked_cluster_ids[0]
#     first_cluster_label = int(
#         next(iter(cluster_report[str(first_cluster_id)].keys()))
#     )

#     return first_cluster_id, first_cluster_label, previous_topk

def process_query(query_embedding, query_content, template_dir, top_k=10):
    """
    新逻辑：
    1. 根据 query 与簇中心的距离，选择最近 top_k 个簇
    2. 返回这些簇中所有日志（带簇标签和与 query 的相似度）
    """
    report_file = os.path.join(
        template_dir,
        f"dbscan_results/{os.path.basename(template_dir)}_cluster_report.json"
    )
    if not os.path.exists(report_file):
        raise FileNotFoundError(f"未找到簇报告文件: {report_file}")

    with open(report_file, "r", encoding="utf-8") as f:
        cluster_report = json.load(f)

    # 所有的 cluster_id（字符串形式）
    all_cluster_ids = sorted(cluster_report.keys(), key=lambda x: int(x))
    if len(all_cluster_ids) == 0:
        raise ValueError("当前模板目录没有任何簇，无法检索日志")

    centers_file = os.path.join(
        template_dir,
        f"dbscan_results/{os.path.basename(template_dir)}_cluster_centers.npy"
    )
    if not os.path.exists(centers_file):
        raise FileNotFoundError(f"未找到簇中心文件: {centers_file}")

    cluster_centers = np.load(centers_file)   # shape: (num_clusters, dim)
    query_emb = np.array(query_embedding).reshape(1, -1)

    distances = norm(cluster_centers - query_emb, axis=1)
    # 得到距离最近的 top_k 个簇的索引
    nearest_idx = np.argsort(distances)[:top_k]

    picked_cluster_ids = [all_cluster_ids[i] for i in nearest_idx]

    logs_file = os.path.join(
        template_dir,
        f"dbscan_results/{os.path.basename(template_dir)}_cluster_top_logs.json"
    )
    if not os.path.exists(logs_file):
        raise FileNotFoundError(f"未找到簇日志文件: {logs_file}")

    with open(logs_file, "r", encoding="utf-8") as f:
        all_logs = json.load(f)

    previous_topk = []

    for cid in picked_cluster_ids:
        str_cid = str(cid)

        # 读取簇日志文本
        logs_file = os.path.join(
            template_dir,
            f"dbscan_results/cluster_{str_cid}_topk_vectors.npy"
        )
        embeddings = np.load(logs_file)  # 对应向量
        json_logs_file = os.path.join(
            template_dir,
            f"dbscan_results/{os.path.basename(template_dir)}_cluster_top_logs.json"
        )
        with open(json_logs_file, "r", encoding="utf-8") as f:
            cluster_top_logs = json.load(f)
        logs_texts = cluster_top_logs[str_cid]

        # 获取簇标签（出现次数最多的）
        cluster_info = cluster_report[str_cid]
        cluster_label = int(max(cluster_info.items(), key=lambda x: x[1])[0])

        # 遍历日志 + embedding
        for text, emb in zip(logs_texts, embeddings):
            previous_topk.append({
                "content": text,
                "embedding": emb,  # numpy array
                "label": cluster_label
            })

    first_cluster_id = picked_cluster_ids[0]
    first_cluster_label = int(
        next(iter(cluster_report[str(first_cluster_id)].keys()))
    )

    return first_cluster_id, first_cluster_label, previous_topk

def read_h5_line(h5_file, n):
    """
    读取 HDF5 中的第 n 行（1-based）
    """
    with h5py.File(h5_file, "r") as f:
        dataset = f["data"]
        if n < 1 or n > len(dataset):
            return None
        line = dataset[n - 1]  # 直接随机访问
        return json.loads(line)


def read_h5_datasets(h5_file, n=10):
    """
    读取HDF5文件中所有数据集的前n行数据
    
    参数:
        h5_file: HDF5文件路径
        n: 要读取的前n行
    
    返回:
        dict: {数据集名: 数据数组}
    """
    with h5py.File(h5_file, 'r') as f:
        result = {}
        for dataset_name in f.keys():
            dataset = f[dataset_name]
            # 读取前n行
            if n < 1 or n > len(dataset):
                return None
            data = dataset[n - 1]
            
            # 如果是字节字符串，解码为普通字符串
            if data.dtype.kind == 'S':
                data = np.array([x.decode('utf-8') for x in data.flatten()]).reshape(data.shape)
            
            result[dataset_name] = data
        
        return result

if __name__ == "__main__":
    tn = fp = fn = tp = 0
    # yuzhi = 0.5
    yuzhi = 1000.65
    for i in range(1):
        in_llm_list = []
        true_labels = []
        pred_labels = []
        in_llm = 0
        # yuzhi = yuzhi + 0.01
        # yuzhi = yuzhi + 1
        for number_of_samples in range(1, BATSH_SIZE):
        # for number_of_samples in [81,89,519,881,981,106,1042,1078,19,1564,1620,1639,1662,1781,121,121,1862,106,280,2048,2103,2179,19,2201,2532,2618,2681,2789,280,2851,2860,106,3029,3174,3246,954,3364,8,3750,106,3797,954,4052,280,4249,954,280,280,106,4795,4848,4899,4932,4933,4969,121,5368,5449,280,5666,5710,5891,6111,6145,6240,4533,954,8,6938,280,280,7205,8,954,8,954,7970,954,8106,8127,8163,8,6938,8542,106,106,106,8823,8933,8948,8967,8,280,9186,9296,9317,9425,106,9567,9586,9728,9737]:
            data_llm = read_h5_datasets('/root/sc_code/bgl/loghub_train_data/all_bgl_zhixing/lora/embeddings/test_embeddings.h5', number_of_samples)
            data_BernoulliNB = read_h5_line('/root/sc_code/bgl/loghub_train_data/all_bgl_zhixing/lora/error_analysis/BernoulliNB.h5', number_of_samples)
            data_CatBoost = read_h5_line('/root/sc_code/bgl/loghub_train_data/all_bgl_zhixing/lora/error_analysis/CatBoost.h5', number_of_samples)
            data_GaussianNB = read_h5_line('/root/sc_code/bgl/loghub_train_data/all_bgl_zhixing/lora/error_analysis/GaussianNB.h5', number_of_samples)
            data_LightGBM = read_h5_line('/root/sc_code/bgl/loghub_train_data/all_bgl_zhixing/lora/error_analysis/LightGBM.h5', number_of_samples)
            data_SGDClassifier = read_h5_line('/root/sc_code/bgl/loghub_train_data/all_bgl_zhixing/lora/error_analysis/SGDClassifier.h5', number_of_samples)
            data_XGBoost = read_h5_line('/root/sc_code/bgl/loghub_train_data/all_bgl_zhixing/lora/error_analysis/XGBoost.h5', number_of_samples)

            model_preds = {
                "BernoulliNB": {
                    "pred": data_BernoulliNB.get('predicted_label'),
                    "conf": data_BernoulliNB.get('confidence') if data_BernoulliNB.get('predicted_label')==1 else 1-data_BernoulliNB.get('confidence')
                },
                "CatBoost": {
                    "pred": data_CatBoost.get('predicted_label'),
                    "conf": data_CatBoost.get('confidence') if data_CatBoost.get('predicted_label')==1 else 1-data_CatBoost.get('confidence')
                },
                "GaussianNB": {
                    "pred": data_GaussianNB.get('predicted_label'),
                    "conf": data_GaussianNB.get('confidence') if data_GaussianNB.get('predicted_label')==1 else 1-data_GaussianNB.get('confidence')
                },
                "LightGBM": {
                    "pred": data_LightGBM.get('predicted_label'),
                    "conf": data_LightGBM.get('confidence') if data_LightGBM.get('predicted_label')==1 else 1-data_LightGBM.get('confidence')
                },
                "SGDClassifier": {
                    "pred": data_SGDClassifier.get('predicted_label'),
                    "conf": data_SGDClassifier.get('confidence') if data_SGDClassifier.get('predicted_label')==1 else 1-data_SGDClassifier.get('confidence')
                },
                "XGBoost": {
                    "pred": data_XGBoost.get('predicted_label'),
                    "conf": data_XGBoost.get('confidence') if data_XGBoost.get('predicted_label')==1 else 1-data_XGBoost.get('confidence')
                }
            }

            f1_list = {
                "BernoulliNB": 0.2791750537501945,
                "CatBoost": 0.9485582862930445,
                "GaussianNB": 0.2155969687479056,
                "LightGBM": 0.7129979411695426,
                "SGDClassifier": 0.8967222066622960,
                "XGBoost": 0.9148990542375236
            }

            score_0 = 0
            score_1 = 0
            for i in model_preds:
                if int(model_preds[i]["pred"]) == 0:
                    score_0 += f1_list[i]*model_preds[i]["conf"]
                else:
                    score_1 += f1_list[i]*model_preds[i]["conf"]

            mb_id = get_mb_folder(data_llm["templates"], MB_DIST)
            if mb_id != 0:
                cluster_id, cluster_label, previous_topk = process_query(data_llm['embeddings'], data_llm['contents'], 
                                                        f"/root/sc_code/bgl/loghub_train_data/all_bgl_zhixing/lora/embeddings/split_by_template_MB/MB{mb_id}", 
                                                        top_k=10)
                query_emb_np = np.array(data_llm["embeddings"]).reshape(1, -1)

                for item in previous_topk:
                    log_emb_np = np.array(item["embedding"]).reshape(1, -1)
                    
                    sim = cosine_similarity(query_emb_np, log_emb_np)[0][0]
                    item["similarity"] = float(sim)
                    # 删除 embedding 字段
                    del item["embedding"]
            else:
                previous_topk = "库中没有相似日志,跟据模型结果自行判断"
            
            # print(f"Sample {number_of_samples}:")
            # print("LLM Embedding:", data_llm['embeddings'])
            # print("LLM Label:", data_llm['labels'])
            # print("LLM Content:", data_llm['contents'])
            # print("LLM Template:", data_llm['templates'])
            # print("LLM MB_DIST ID:", MB_DIST[str(data_llm['templates'])])
            # print(previous_topk)
            # print(f"Cluster ID: {cluster_id}")
            # print(f"Cluster Label: {cluster_label}")
            # print("Previous Top-k Logs:")
            # for log in previous_topk:
            #     print(log)
            # print("BernoulliNB Prediction:", data_BernoulliNB.get('predicted_label'))
            # print("BernoulliNB Confidence:", data_BernoulliNB.get('confidence'))
            
            # print("CatBoost Prediction:", data_CatBoost.get('predicted_label'))
            # print("CatBoost Confidence:", data_CatBoost.get('confidence'))

            # print("GaussianNB Prediction:", data_GaussianNB.get('predicted_label'))
            # print("GaussianNB Confidence:", data_GaussianNB.get('confidence'))

            # print("LightGBM Prediction:", data_LightGBM.get('predicted_label'))
            # print("LightGBM Confidence:", data_LightGBM.get('confidence'))

            # print("SGDClassifier Prediction:", data_SGDClassifier.get('predicted_label'))
            # print("SGDClassifier Confidence:", data_SGDClassifier.get('confidence'))

            # print("XGBoost Prediction:", data_XGBoost.get('predicted_label'))
            # print("XGBoost Confidence:", data_XGBoost.get('confidence'))
            # print("-" * 50)
            
            if score_0 > score_1 + yuzhi:
                # pred_labels.append(0)
                # true_labels.append(int(data_llm['labels']))
                pred_label = 0
            elif score_1 > score_0 + yuzhi:
                # pred_labels.append(1)
                # true_labels.append(int(data_llm['labels']))
                pred_label = 1
            else:
                # if score_0 > score_1:
                #     # pred_labels.append(0)
                #     pred_label = 0
                # else:
                #     # pred_labels.append(1)
                #     pred_label = 1

                in_llm += 1
                in_llm_list.append(number_of_samples)
                # continue

                result = deepseek_rag_inference(
                    llm_content=data_llm['contents'].item(),
                    topk_logs=previous_topk,
                    model_preds=model_preds
                )
                # true_labels.append(int(data_llm['labels']))
                # pred_labels.append(int(result['pred_label']))
                pred_label = int(result['pred_label'])
                if int(result['pred_label']) != int(data_llm['labels']):
                    error_record = {
                        "true_label": int(data_llm['labels']),
                        "pred_label": int(result['pred_label']),
                        "llm_content": data_llm['contents'].item(),
                        "topk_logs": previous_topk,
                        "model_preds": model_preds,
                        "reson": result.get("reasoning", "")
                    }
                    with open("llm_wrong_predictions.jsonl_1213", "a", encoding="utf-8") as f:
                        f.write(json.dumps(error_record, ensure_ascii=False) + "\n")

            true_label = int(data_llm['labels'])
            if pred_label == 0:
                if true_label == 0:
                    tn += 1
                else:
                    fn += 1
            else:  # pred_label == 1
                if true_label == 0:
                    fp += 1
                else:
                    tp += 1
            
            if number_of_samples%1000 ==0:
                print(number_of_samples//1000)

        # tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()

        # 计算指标
        # accuracy = accuracy_score(true_labels, pred_labels)
        # precision = precision_score(true_labels, pred_labels)
        # recall = recall_score(true_labels, pred_labels)
        # f1 = f1_score(true_labels, pred_labels)

        total = tn + fp + fn + tp

        if total > 0:
            accuracy = (tp + tn) / total
        else:
            accuracy = 0
            print("警告: 总样本数为0")

        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
            print("警告: 预测为正例的样本数为0")

        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
            print("警告: 实际为正例的样本数为0")

        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
            print("警告: 精确率和召回率都为0")


        print("=== LLM 预测指标 ===")
        print(f"样本总数: {len(true_labels)}")
        print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(in_llm)
        print(in_llm_list)
        print(yuzhi)

        json_file_path = "llm_metrics_1212.json"
        def save_metrics_to_json(metrics_data, file_path=json_file_path):
            """将指标保存为JSON格式，处理NumPy类型"""
            import numpy as np
            
            def convert_numpy_types(obj):
                """递归转换NumPy数据类型为Python原生类型"""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            # 转换NumPy类型
            metrics_data_converted = convert_numpy_types(metrics_data)
            
            # 读取现有数据
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_metrics = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                all_metrics = []
            
            # 添加新数据
            all_metrics.append(metrics_data_converted)
            
            # 保存回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, ensure_ascii=False, indent=2)


        # 在计算完指标后，构建 metrics_data 之前，确保所有值是 Python 原生类型
        # 直接转换为 Python 类型
        tp, fp, tn, fn = int(tp), int(fp), int(tn), int(fn)
        accuracy = float(accuracy)
        precision = float(precision)
        recall = float(recall)
        f1 = float(f1)

        # 确保 in_llm 是 Python int
        in_llm = int(in_llm)

        # 构建 metrics_data - 现在所有值都是 Python 原生类型
        metrics_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "循环信息": {
                "样本总数": len(true_labels),
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn
            },
            "性能指标": {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            },
            "输入信息": {
                "in_llm": in_llm,
                "in_llm_list": in_llm_list  # 确保 in_llm_list 中的元素也是 Python 原生类型
            }
        }

        # 保存到 JSON
        save_metrics_to_json(metrics_data)


    